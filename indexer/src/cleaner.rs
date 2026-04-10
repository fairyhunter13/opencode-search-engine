//! Automatic cleanup of orphaned and stale project embedding directories.
//!
//! Scans `~/.local/share/opencode/projects/` and removes:
//! - Orphaned projects (source directory no longer exists)
//! - Stale projects (not modified in N days, configurable)
//! - Auxiliary dirs (backups, compaction-history, corrupted-backup-*)

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

// Default interval: 24 hours
const DEFAULT_INTERVAL_SECS: u64 = 86400;
// Default stale threshold: 90 days
const DEFAULT_STALE_DAYS: u64 = 90;

/// Configuration for the cleaner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// How often to run cleanup (seconds)
    pub interval: u64,
    /// Days after which unused project dirs are considered stale
    pub stale_days: u64,
    /// Whether to clean orphaned project dirs
    pub orphans: bool,
    /// Whether to clean stale project dirs
    pub stale: bool,
    /// Whether to clean auxiliary dirs (backups, compaction-history, etc.)
    pub aux: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            interval: DEFAULT_INTERVAL_SECS,
            stale_days: DEFAULT_STALE_DAYS,
            orphans: true,
            stale: true,
            aux: true,
        }
    }
}

/// Read config from env vars, falling back to defaults.
pub fn config() -> Config {
    Config {
        interval: std::env::var("OPENCODE_CLEANUP_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_INTERVAL_SECS),
        stale_days: std::env::var("OPENCODE_CLEANUP_STALE_DAYS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_STALE_DAYS),
        orphans: std::env::var("OPENCODE_CLEANUP_ORPHANS")
            .map(|v| v != "0" && v != "false")
            .unwrap_or(true),
        stale: std::env::var("OPENCODE_CLEANUP_STALE")
            .map(|v| v != "0" && v != "false")
            .unwrap_or(true),
        aux: std::env::var("OPENCODE_CLEANUP_AUX")
            .map(|v| v != "0" && v != "false")
            .unwrap_or(true),
    }
}

/// Info about a project directory discovered during scan.
#[derive(Debug, Clone, Serialize)]
pub struct ProjectInfo {
    /// The project directory path (e.g. ~/.local/share/opencode/projects/<id>)
    pub path: PathBuf,
    /// The project ID (directory name)
    pub id: String,
    /// The project root from .symlinks.json (if available)
    pub root: Option<String>,
    /// Size in bytes
    pub bytes: u64,
    /// Last modification time
    pub modified: Option<SystemTime>,
}

/// Classification of why a project should be cleaned up.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum Reason {
    /// Project root directory no longer exists
    Orphaned,
    /// Project hasn't been modified in stale_days
    Stale,
}

/// A project identified for cleanup.
#[derive(Debug, Clone, Serialize)]
pub struct Target {
    pub info: ProjectInfo,
    pub reason: Reason,
}

/// Result of a cleanup operation.
#[derive(Debug, Clone, Serialize, Default)]
pub struct Report {
    /// Number of orphaned project dirs removed
    pub orphans: u64,
    /// Number of stale project dirs removed
    pub stale: u64,
    /// Number of auxiliary dirs removed
    pub aux_dirs: u64,
    /// Total bytes freed
    pub freed: u64,
    /// Errors encountered (non-fatal)
    pub errors: Vec<String>,
    /// Details of what was cleaned (for dry_run)
    pub targets: Vec<Target>,
}

/// Compute directory size recursively (expensive — only used by 24h fallback path).
fn dir_size(path: &Path) -> u64 {
    walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// Get the top-level mtime of a directory via a single stat() call.
/// Fast O(1) replacement for the recursive latest_modified used in normal cleanup.
fn dir_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// Get the latest modification time of any file in a directory (full walkdir).
/// Only used when stale detection needs deep mtime accuracy (24h fallback path).
fn latest_modified(path: &Path) -> Option<SystemTime> {
    walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok().and_then(|m| m.modified().ok()))
        .max()
}

/// Read project_root from .symlinks.json in a project directory.
fn read_root(dir: &Path) -> Option<String> {
    let file = dir.join(".symlinks.json");
    let data = std::fs::read_to_string(&file).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    json.get("project_root")?.as_str().map(String::from)
}

/// Cursor file path for batched scan resumption.
fn cursor_path(base: &Path) -> std::path::PathBuf {
    // Write cursor OUTSIDE projects/ to avoid triggering inotify on projects/
    base.join("cleaner_cursor")
}

/// Read the current cursor offset from disk (0 if absent/invalid).
fn read_cursor(base: &Path) -> usize {
    std::fs::read_to_string(cursor_path(base))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Write the cursor offset to disk.
fn write_cursor(base: &Path, offset: usize) {
    let _ = std::fs::write(cursor_path(base), offset.to_string());
}

/// Batch size for each scan invocation (default 500, env OPENCODE_INDEXER_CLEANUP_BATCH).
fn batch_size() -> usize {
    std::env::var("OPENCODE_INDEXER_CLEANUP_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500)
}

/// Scan all project directories under the shared data dir.
/// Uses shallow stat (O(1) per dir) instead of recursive walkdir.
/// Called by identify() which handles the orphan/stale logic.
pub fn scan(base: &Path) -> Vec<ProjectInfo> {
    scan_full(base, false)
}

/// Full scan using recursive walkdir for deep mtime/size accuracy.
/// Only called by the 24h fallback path.
pub fn scan_deep(base: &Path) -> Vec<ProjectInfo> {
    scan_full(base, true)
}

fn scan_full(base: &Path, deep: bool) -> Vec<ProjectInfo> {
    let dir = base.join("projects");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return vec![];
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| {
            let path = e.path();
            let id = e.file_name().to_string_lossy().to_string();
            let root = read_root(&path);
            // Use cheap O(1) stat for event-driven path; deep walkdir only for 24h pass
            let (bytes, modified) = if deep {
                (dir_size(&path), latest_modified(&path))
            } else {
                (0u64, dir_mtime(&path))
            };
            ProjectInfo {
                path,
                id,
                root,
                bytes,
                modified,
            }
        })
        .collect()
}

/// Batched scan: process at most `batch_size` entries per call, resuming via cursor.
/// Returns (infos, complete) where complete=true when a full pass has finished.
pub fn scan_batch(base: &Path) -> (Vec<ProjectInfo>, bool) {
    let dir = base.join("projects");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return (vec![], true);
    };
    // Collect and sort IDs for stable cursor ordering
    let mut ids: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    ids.sort();

    let total = ids.len();
    let cursor = read_cursor(base).min(total);
    let batch = batch_size();
    let end = (cursor + batch).min(total);
    let complete = end >= total;
    let next = if complete { 0 } else { end };
    write_cursor(base, next);

    let infos: Vec<ProjectInfo> = ids[cursor..end]
        .iter()
        .map(|id| {
            let path = dir.join(id);
            let root = read_root(&path);
            let modified = dir_mtime(&path);
            ProjectInfo {
                path,
                id: id.clone(),
                root,
                bytes: 0,
                modified,
            }
        })
        .collect();

    (infos, complete)
}

/// Identify projects that should be cleaned up.
pub fn identify(projects: &[ProjectInfo], cfg: &Config) -> Vec<Target> {
    let threshold = SystemTime::now() - Duration::from_secs(cfg.stale_days * 86400);
    let mut targets = vec![];

    for info in projects {
        // Check orphaned: project_root doesn't exist
        if cfg.orphans {
            if let Some(ref root) = info.root {
                if !Path::new(root).exists() {
                    targets.push(Target {
                        info: info.clone(),
                        reason: Reason::Orphaned,
                    });
                    continue;
                }
            }
        }

        // Check stale: not modified recently
        if cfg.stale {
            if let Some(modified) = info.modified {
                if modified < threshold {
                    targets.push(Target {
                        info: info.clone(),
                        reason: Reason::Stale,
                    });
                }
            }
        }
    }

    targets
}

/// Find auxiliary directories that can be cleaned.
fn aux_dirs(base: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![];
    // backups/
    let backups = base.join("backups");
    if backups.is_dir() {
        dirs.push(backups);
    }
    // compaction-history/
    let compaction = base.join("compaction-history");
    if compaction.is_dir() {
        dirs.push(compaction);
    }
    // corrupted-backup-* dirs
    if let Ok(entries) = std::fs::read_dir(base) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("corrupted-backup-") && entry.path().is_dir() {
                dirs.push(entry.path());
            }
        }
    }
    dirs
}

/// Execute cleanup using batched shallow scan (event-driven path).
/// Returns (report, complete) where complete=true after a full pass.
pub fn run_batch(base: &Path, cfg: &Config, dry: bool) -> (Report, bool) {
    let mut report = Report::default();
    let (projects, complete) = scan_batch(base);
    let targets = identify(&projects, cfg);

    for target in &targets {
        // bytes=0 in batch path (shallow scan) — compute only when deleting
        let size = if !dry { dir_size(&target.info.path) } else { 0 };
        if !dry {
            if let Err(e) = std::fs::remove_dir_all(&target.info.path) {
                report.errors.push(format!(
                    "failed to remove {}: {}",
                    target.info.path.display(),
                    e
                ));
                continue;
            }
        }
        report.freed += size;
        match target.reason {
            Reason::Orphaned => report.orphans += 1,
            Reason::Stale => report.stale += 1,
        }
    }
    report.targets = targets;
    (report, complete)
}

/// Execute full cleanup: remove orphaned/stale project dirs and auxiliary dirs.
/// Uses deep recursive scan — intended for the 24h fallback timer path.
/// If `dry` is true, only report what would be cleaned without deleting.
pub fn run(base: &Path, cfg: &Config, dry: bool) -> Report {
    let mut report = Report::default();
    let projects = scan_deep(base);
    let targets = identify(&projects, cfg);

    for target in &targets {
        let size = target.info.bytes;
        if !dry {
            if let Err(e) = std::fs::remove_dir_all(&target.info.path) {
                report.errors.push(format!(
                    "failed to remove {}: {}",
                    target.info.path.display(),
                    e
                ));
                continue;
            }
        }
        report.freed += size;
        match target.reason {
            Reason::Orphaned => report.orphans += 1,
            Reason::Stale => report.stale += 1,
        }
    }
    report.targets = targets;

    // Clean auxiliary dirs
    if cfg.aux {
        for dir in aux_dirs(base) {
            let size = dir_size(&dir);
            if !dry {
                if let Err(e) = std::fs::remove_dir_all(&dir) {
                    report
                        .errors
                        .push(format!("failed to remove {}: {}", dir.display(), e));
                    continue;
                }
            }
            report.aux_dirs += 1;
            report.freed += size;
        }
    }

    report
}

/// Duration for the periodic cleanup interval, reading from env.
pub fn interval() -> Duration {
    Duration::from_secs(config().interval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup() -> (TempDir, PathBuf) {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().to_path_buf();
        fs::create_dir_all(base.join("projects")).unwrap();
        (tmp, base)
    }

    fn make_project(base: &Path, id: &str, root: Option<&str>) -> PathBuf {
        let dir = base.join("projects").join(id);
        fs::create_dir_all(&dir).unwrap();
        if let Some(root) = root {
            let json = serde_json::json!({
                "version": 1,
                "updated_at": 0,
                "project_root": root,
                "symlinks": []
            });
            fs::write(dir.join(".symlinks.json"), json.to_string()).unwrap();
        }
        // Add a dummy file so dir_size > 0
        fs::write(dir.join("data.bin"), vec![0u8; 1024]).unwrap();
        dir
    }

    #[test]
    fn test_scan_empty() {
        let (_tmp, base) = setup();
        let projects = scan(&base);
        assert!(projects.is_empty());
    }

    #[test]
    fn test_scan_finds_projects() {
        let (_tmp, base) = setup();
        make_project(&base, "abc123", Some("/tmp/existing"));
        make_project(&base, "def456", None);
        let projects = scan(&base);
        assert_eq!(projects.len(), 2);
    }

    #[test]
    fn test_identify_orphaned() {
        let (_tmp, base) = setup();
        // Point to a non-existent directory
        make_project(&base, "orphan1", Some("/nonexistent/path/12345"));
        let projects = scan(&base);
        let cfg = Config::default();
        let targets = identify(&projects, &cfg);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].reason, Reason::Orphaned);
        assert_eq!(targets[0].info.id, "orphan1");
    }

    #[test]
    fn test_identify_not_orphaned() {
        let (tmp, base) = setup();
        // Point to a directory that exists (the temp dir itself)
        make_project(&base, "valid1", Some(tmp.path().to_str().unwrap()));
        let projects = scan(&base);
        let cfg = Config::default();
        let targets = identify(&projects, &cfg);
        assert!(targets.is_empty());
    }

    #[test]
    fn test_identify_stale() {
        let (_tmp, base) = setup();
        let dir = make_project(&base, "stale1", None);
        // Set modification time to 100 days ago on ALL files and the dir itself
        let old = filetime::FileTime::from_system_time(
            SystemTime::now() - Duration::from_secs(100 * 86400),
        );
        for entry in walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let _ = filetime::set_file_mtime(entry.path(), old);
        }
        filetime::set_file_mtime(&dir, old).unwrap();
        let projects = scan(&base);
        let cfg = Config {
            stale_days: 90,
            ..Config::default()
        };
        let targets = identify(&projects, &cfg);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].reason, Reason::Stale);
    }

    #[test]
    fn test_identify_not_stale() {
        let (_tmp, base) = setup();
        // Fresh project — just created, not stale
        make_project(&base, "fresh1", None);
        let projects = scan(&base);
        let cfg = Config {
            stale_days: 90,
            ..Config::default()
        };
        let targets = identify(&projects, &cfg);
        assert!(targets.is_empty());
    }

    #[test]
    fn test_run_dry() {
        let (_tmp, base) = setup();
        make_project(&base, "orphan1", Some("/nonexistent/12345"));
        let cfg = Config::default();
        let report = run(&base, &cfg, true);
        assert_eq!(report.orphans, 1);
        assert!(report.freed > 0);
        // Dir should still exist (dry run)
        assert!(base.join("projects").join("orphan1").exists());
    }

    #[test]
    fn test_run_delete() {
        let (_tmp, base) = setup();
        make_project(&base, "orphan1", Some("/nonexistent/12345"));
        let cfg = Config::default();
        let report = run(&base, &cfg, false);
        assert_eq!(report.orphans, 1);
        assert!(report.freed > 0);
        // Dir should be gone
        assert!(!base.join("projects").join("orphan1").exists());
    }

    #[test]
    fn test_aux_dirs() {
        let (_tmp, base) = setup();
        fs::create_dir_all(base.join("backups")).unwrap();
        fs::write(base.join("backups").join("backup.db"), vec![0u8; 512]).unwrap();
        fs::create_dir_all(base.join("compaction-history")).unwrap();
        fs::create_dir_all(base.join("corrupted-backup-20260206")).unwrap();
        let cfg = Config::default();
        let report = run(&base, &cfg, false);
        assert_eq!(report.aux_dirs, 3);
        assert!(!base.join("backups").exists());
        assert!(!base.join("compaction-history").exists());
        assert!(!base.join("corrupted-backup-20260206").exists());
    }

    #[test]
    fn test_config_disabled() {
        let (_tmp, base) = setup();
        make_project(&base, "orphan1", Some("/nonexistent/12345"));
        fs::create_dir_all(base.join("backups")).unwrap();
        let cfg = Config {
            orphans: false,
            stale: false,
            aux: false,
            ..Config::default()
        };
        let report = run(&base, &cfg, false);
        assert_eq!(report.orphans, 0);
        assert_eq!(report.aux_dirs, 0);
        // Everything should still exist
        assert!(base.join("projects").join("orphan1").exists());
        assert!(base.join("backups").exists());
    }

    #[test]
    fn test_orphan_takes_priority_over_stale() {
        let (_tmp, base) = setup();
        let dir = make_project(&base, "both1", Some("/nonexistent/12345"));
        // Also make it stale
        let old = filetime::FileTime::from_system_time(
            SystemTime::now() - Duration::from_secs(100 * 86400),
        );
        filetime::set_file_mtime(dir.join("data.bin"), old).unwrap();
        let projects = scan(&base);
        let cfg = Config::default();
        let targets = identify(&projects, &cfg);
        // Should only appear once, as orphaned (not duplicate as stale)
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].reason, Reason::Orphaned);
    }
}
