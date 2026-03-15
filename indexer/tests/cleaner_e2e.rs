//! End-to-end integration test for the cleaner module.
//!
//! Tests the full cleanup lifecycle: scan → identify → clean → verify.

use std::fs;
use std::path::Path;

use opencode_indexer::cleaner;

fn make_project(base: &Path, id: &str, root: Option<&str>) {
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
    // Create .lancedb subdirectory with some data
    let lance = dir.join(".lancedb");
    fs::create_dir_all(&lance).unwrap();
    fs::write(lance.join("chunks.dat"), vec![0u8; 4096]).unwrap();
    // Create memories subdirectory
    let mem = dir.join("memories");
    fs::create_dir_all(&mem).unwrap();
    fs::write(mem.join("note.md"), "# Memory\nSome content here").unwrap();
}

#[test]
fn e2e_full_cleanup_lifecycle() {
    let tmp = tempfile::TempDir::new().unwrap();
    let base = tmp.path();

    // Create a "valid" project root that exists
    let valid_root = base.join("source-code");
    fs::create_dir_all(&valid_root).unwrap();

    // Setup projects directory
    fs::create_dir_all(base.join("projects")).unwrap();

    // Create 3 orphaned projects (point to non-existent paths)
    make_project(base, "orphan_aaa", Some("/nonexistent/project/aaa"));
    make_project(base, "orphan_bbb", Some("/nonexistent/project/bbb"));
    make_project(base, "orphan_ccc", Some("/tmp/definitely-not-here-12345"));

    // Create 2 valid projects (point to existing paths)
    make_project(base, "valid_111", Some(valid_root.to_str().unwrap()));
    make_project(base, "valid_222", Some(base.to_str().unwrap()));

    // Create 1 project with no symlinks.json (should not be orphaned, might be stale)
    make_project(base, "nosymlinks", None);

    // Create auxiliary dirs
    fs::create_dir_all(base.join("backups")).unwrap();
    fs::write(base.join("backups").join("old.db"), vec![0u8; 2048]).unwrap();
    fs::create_dir_all(base.join("compaction-history")).unwrap();
    fs::write(base.join("compaction-history").join("entry.json"), "{}").unwrap();
    fs::create_dir_all(base.join("corrupted-backup-20260206-080243")).unwrap();
    fs::write(
        base.join("corrupted-backup-20260206-080243")
            .join("corrupt.db"),
        vec![0u8; 1024],
    )
    .unwrap();

    // Step 1: Dry run — nothing should be deleted
    let cfg = cleaner::Config {
        stale_days: 90,
        orphans: true,
        stale: true,
        aux: true,
        ..cleaner::Config::default()
    };

    let dry = cleaner::run(base, &cfg, true);
    assert_eq!(dry.orphans, 3, "dry run should find 3 orphans");
    assert_eq!(dry.aux_dirs, 3, "dry run should find 3 aux dirs");
    assert!(dry.freed > 0, "dry run should report bytes to free");
    assert!(dry.errors.is_empty(), "dry run should have no errors");
    // Verify nothing was actually deleted
    assert!(base.join("projects").join("orphan_aaa").exists());
    assert!(base.join("projects").join("orphan_bbb").exists());
    assert!(base.join("projects").join("orphan_ccc").exists());
    assert!(base.join("backups").exists());

    // Step 2: Real run — orphans and aux dirs should be deleted
    let report = cleaner::run(base, &cfg, false);
    assert_eq!(report.orphans, 3, "should remove 3 orphans");
    assert_eq!(report.stale, 0, "no stale projects (all freshly created)");
    assert_eq!(report.aux_dirs, 3, "should remove 3 aux dirs");
    assert!(report.freed > 0, "should free bytes");
    assert!(
        report.errors.is_empty(),
        "should have no errors: {:?}",
        report.errors
    );

    // Verify orphans are gone
    assert!(!base.join("projects").join("orphan_aaa").exists());
    assert!(!base.join("projects").join("orphan_bbb").exists());
    assert!(!base.join("projects").join("orphan_ccc").exists());

    // Verify valid projects still exist
    assert!(base.join("projects").join("valid_111").exists());
    assert!(base.join("projects").join("valid_222").exists());
    assert!(base.join("projects").join("nosymlinks").exists());

    // Verify .lancedb content preserved in valid projects
    assert!(base
        .join("projects")
        .join("valid_111")
        .join(".lancedb")
        .join("chunks.dat")
        .exists());

    // Verify aux dirs are gone
    assert!(!base.join("backups").exists());
    assert!(!base.join("compaction-history").exists());
    assert!(!base.join("corrupted-backup-20260206-080243").exists());

    // Step 3: Run again — nothing to clean
    let report2 = cleaner::run(base, &cfg, false);
    assert_eq!(report2.orphans, 0);
    assert_eq!(report2.stale, 0);
    assert_eq!(report2.aux_dirs, 0);
    assert_eq!(report2.freed, 0);
}

#[test]
fn e2e_cleanup_with_disabled_features() {
    let tmp = tempfile::TempDir::new().unwrap();
    let base = tmp.path();

    fs::create_dir_all(base.join("projects")).unwrap();
    make_project(base, "orphan_x", Some("/nonexistent/x"));
    fs::create_dir_all(base.join("backups")).unwrap();

    // Disable orphan and aux cleanup
    let cfg = cleaner::Config {
        orphans: false,
        stale: false,
        aux: false,
        ..cleaner::Config::default()
    };

    let report = cleaner::run(base, &cfg, false);
    assert_eq!(report.orphans, 0);
    assert_eq!(report.aux_dirs, 0);
    // Everything should still exist
    assert!(base.join("projects").join("orphan_x").exists());
    assert!(base.join("backups").exists());
}

#[test]
fn e2e_scan_reports_correct_sizes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let base = tmp.path();

    fs::create_dir_all(base.join("projects")).unwrap();
    make_project(base, "sized_proj", Some("/nonexistent/sized"));

    let projects = cleaner::scan(base);
    assert_eq!(projects.len(), 1);
    // .lancedb/chunks.dat (4096) + memories/note.md (~24) + .symlinks.json (~80) > 4000
    assert!(
        projects[0].bytes > 4000,
        "project size should include all files, got {}",
        projects[0].bytes
    );
    assert_eq!(projects[0].id, "sized_proj");
    assert_eq!(projects[0].root.as_deref(), Some("/nonexistent/sized"));
}
