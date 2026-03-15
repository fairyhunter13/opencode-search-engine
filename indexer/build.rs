// Build script to generate version info at compile time
// Version format: YYYY-MM-DD-<git-sha> (matching TypeScript build)

use std::process::Command;

fn main() {
    // Get git commit date in YYYY-MM-DD format (matches TypeScript build for consistency)
    let date = Command::new("git")
        .args(["log", "-1", "--format=%cs", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Get git commit hash (short)
    let commit = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Set combined version string
    let version = format!("{}-{}", date, commit);
    println!("cargo:rustc-env=OPENCODE_VERSION={}", version);

    // Rerun if git HEAD changes
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads/");
}
