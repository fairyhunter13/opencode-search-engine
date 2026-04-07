---
name: git-sync-default-rebase-force
description: "Sync local+origin default branch to upstream, rebase ALL feature branches onto default, and recursively sync submodule forks. Smart conflict resolution with mandatory source code analysis."
---

# Skill: git-sync-default-rebase-force

Sync local+origin default branch to upstream, rebase ALL feature branches onto default, and recursively sync submodule forks. Auto-detects default branch name.

## Use this when

- Your repo is a fork with `origin` (fork) and `upstream` (source)
- You need `origin/<default>` to match `upstream/<default>`
- You want to rebase ALL feature branches onto the updated default branch
- You have submodules that are also forks and need syncing

## Assumptions

- Remotes are named `origin` and `upstream`
- Use `--force-with-lease` (preferred over `--force`)

## Steps

### 0. Auto-detect default branch

```bash
# Detect the default branch name from upstream (or origin if no upstream)
if git remote | grep -q upstream; then
  DEFAULT=$(git remote show upstream 2>/dev/null | grep "HEAD branch" | awk '{print $NF}')
else
  DEFAULT=$(git remote show origin 2>/dev/null | grep "HEAD branch" | awk '{print $NF}')
fi
# Fallback: check common names
if [ -z "$DEFAULT" ]; then
  for branch in dev main master; do
    if git rev-parse --verify "origin/$branch" >/dev/null 2>&1; then
      DEFAULT=$branch; break
    fi
  done
fi
echo "Default branch: $DEFAULT"
```

Use `$DEFAULT` in place of hardcoded `dev` in all subsequent steps.

### 1. Fetch latest refs

```bash
git fetch --prune origin
git remote | grep -q upstream && git fetch --prune upstream
```

### 2. Sync default branch locally with upstream

**Skip this step if there is no `upstream` remote (owned repo, not a fork).**

```bash
git switch $DEFAULT
git merge --ff-only upstream/$DEFAULT
```

If `--ff-only` fails because `$DEFAULT` diverged, you are about to overwrite history. To hard-sync to upstream:

```bash
git reset --hard upstream/$DEFAULT
```

### 3. Sync default branch on your fork (`origin`)

```bash
git push origin $DEFAULT
```

If you hard-synced with `reset --hard` and `origin/$DEFAULT` diverged, you may need:

```bash
git push --force-with-lease origin $DEFAULT
```

**Note:** When syncing the default branch to upstream, you may use `--no-verify` if pre-push hooks fail due to issues in upstream code (not your changes). This is acceptable since you're just mirroring upstream's state.

4. Rebase current branch onto updated default

```bash
git switch -
git rebase $DEFAULT
```

If conflicts occur, follow the **Mandatory Conflict Resolution** procedure below before continuing.

After all conflicts are resolved and rebase completes:

```bash
git push --force-with-lease
```

**Note:** If pre-push hooks fail due to upstream typecheck issues (errors that also exist on `$DEFAULT`), you may use `--no-verify`. But if hooks fail due to YOUR code errors (missing functions, broken references), that indicates conflict resolution dropped your changes -- see **Post-Resolution Verification**.

### 5. Sync ALL feature branches onto updated default

After syncing the default branch, rebase and force-push ALL other local feature branches that track origin:

```bash
ORIGINAL=$(git branch --show-current)
for branch in $(git for-each-ref --format='%(refname:short)' refs/heads/ | grep -v "^$DEFAULT$"); do
  echo "=== Rebasing $branch ==="
  git switch "$branch"
  git rebase $DEFAULT
  git push --force-with-lease --no-verify
done
git switch "$ORIGINAL"
```

If any branch has conflicts, follow the **Mandatory Conflict Resolution** procedure below before continuing to the next branch.

### 6. Sync submodule forks (if this is a parent repo with submodules)

For each submodule that is a fork (has an `upstream` remote), recursively apply the same git-sync-default-rebase-force procedure:

```bash
git submodule foreach --recursive '
  if git remote | grep -q upstream; then
    echo "=== Syncing fork: $name ==="
    DEFAULT=$(git remote show upstream 2>/dev/null | grep "HEAD branch" | awk "{print \$NF}")
    [ -z "$DEFAULT" ] && DEFAULT=main
    git fetch --prune origin
    git fetch --prune upstream
    CURRENT=$(git branch --show-current)
    git switch $DEFAULT
    git merge --ff-only upstream/$DEFAULT || git reset --hard upstream/$DEFAULT
    git push origin $DEFAULT --no-verify
    git switch "$CURRENT"
    git rebase $DEFAULT
    git push --force-with-lease --no-verify
  fi
'
```

After syncing submodules, update the parent repo's submodule pointers if they changed:

```bash
git add -u
if ! git diff --cached --quiet; then
  git commit --amend --no-edit
  git push --force-with-lease --no-verify
fi
```

## Mandatory Conflict Resolution

**CRITICAL RULE: NEVER blindly use `git checkout --ours` or `git checkout --theirs` on source code files.** This drops ALL changes from one side, not just the conflicted section. It is only safe for files where one side's version is entirely correct (e.g., version bumps, lockfiles).

**PRESERVATION RULE: Your branch's changes MUST be preserved.** Before starting conflict resolution, capture what your branch changed so you can verify nothing was lost:

```bash
# Save ALL your branch's changes vs dev for reference
git diff $DEFAULT...HEAD --stat           # Overview of what files you changed
git diff $DEFAULT...HEAD > /tmp/my-branch-changes.patch   # Full diff for verification
```

When `git rebase $DEFAULT` encounters conflicts, you MUST follow this procedure for EVERY conflicted file.

### Step 1: Triage -- Classify Each File

List all conflicted files and classify them:

```bash
git diff --name-only --diff-filter=U
```

Classify each file into one of these categories:

| Category                    | Examples                                     | Safe Resolution                                                       |
| --------------------------- | -------------------------------------------- | --------------------------------------------------------------------- |
| **Trivial: version bump**   | `package.json` (only version field differs)  | `git checkout --ours <file>` (take dev's version)                     |
| **Trivial: lockfile**       | `bun.lock`, `package-lock.json`, `yarn.lock` | `git checkout --ours <file>` (take dev's, regenerate later if needed) |
| **Trivial: config version** | `extension.toml` (only version/URL differs)  | `git checkout --ours <file>` (take dev's version)                     |
| **Source code**             | `.ts`, `.tsx`, `.js`, `.go`, etc.            | **MUST use Smart Analysis** (see below)                               |
| **Test files**              | `*.test.ts`, `*.spec.ts`                     | **MUST use Smart Analysis**                                           |

**The key distinction:** Trivial files have no logic -- one side is entirely correct. Source code files may have changes from BOTH sides that must be preserved.

### Step 2: Resolve Trivial Files First

Handle all trivial files in batch:

```bash
# Version bumps and lockfiles -- take dev's version (--ours in rebase context)
git checkout --ours bun.lock
git checkout --ours packages/*/package.json
git checkout --ours sdks/*/package.json
# ... etc for all trivial files
```

### Step 3: Smart Analysis for Source Code Files (MANDATORY)

**This step is NON-NEGOTIABLE for all source code files. Skipping Smart Analysis WILL result in lost changes.**

For EACH source code file with conflicts, you MUST:

#### 3a. Understand both sides

```bash
# See the conflict markers and their context
git diff <file>

# See what YOUR branch changed (theirs in rebase context)
git diff :1:<file> :3:<file>

# See what UPSTREAM changed (ours in rebase context)
git diff :1:<file> :2:<file>
```

**IMPORTANT -- Rebase direction reminder:**

- In a rebase, `--ours` / `:2:` = the branch being rebased ONTO (dev/upstream)
- In a rebase, `--theirs` / `:3:` = the commit being replayed (YOUR branch)
- This is the OPPOSITE of a normal merge!

#### 3b. Measure the scope of each side's changes

```bash
# Count lines changed by your branch
git diff :1:<file> :3:<file> | wc -l

# Count lines changed by upstream
git diff :1:<file> :2:<file> | wc -l
```

If your branch has significantly more changes (e.g., you added hundreds of lines of new code), you likely want to start from YOUR version (`--theirs`) and integrate upstream's smaller changes on top.

If upstream has significantly more changes, start from UPSTREAM (`--ours`) and replay your changes on top.

#### 3c. Categorize upstream changes as conflicting vs non-conflicting (MANDATORY)

**Before choosing a strategy, you MUST classify every upstream change:**

Read upstream's full diff (`git diff :1:<file> :2:<file>`) and for EACH change, determine:

| Category                        | Description                                                                                                                | Action                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Non-conflicting**             | Upstream changed code your branch did NOT touch (different functions, different regions)                                   | Safe to integrate                                                           |
| **Conflicting — same region**   | Both sides modified the same lines/block                                                                                   | Must choose one side or manually merge                                      |
| **Conflicting — architectural** | Upstream restructured/removed code that your branch builds on (e.g., upstream replaced a component your branch customized) | Your version wins — upstream's restructuring is incompatible with your work |

**CRITICAL: If ALL of upstream's changes are conflicting (especially architectural conflicts), taking your version wholesale IS the correct Smart Analysis outcome.** Do NOT force-integrate upstream changes that would destroy your custom work. Document why in a comment before staging.

Example of architectural conflict: upstream extracts a component into a plugin slot, but your branch has a fully custom version of that component. Adopting the plugin slot would delete your custom work.

#### 3d. Choose resolution strategy

**DEFAULT BIAS: When in doubt, favor preserving YOUR changes.** Your branch's work is the primary goal; upstream can be re-merged, but your work should not be lost.

**Strategy A: Start from your version (PREFERRED when your changes are larger or critical)**

```bash
git checkout --theirs <file>    # Start from YOUR version
```

Then, for EACH non-conflicting upstream change identified in 3c, manually apply it on top:

```bash
# Read upstream's diff to find non-conflicting changes
git diff :1:<file> :2:<file>
# Apply ONLY the non-conflicting hunks manually via Edit
```

**If there are ZERO non-conflicting upstream changes, your version is the final result. No further edits needed.**

Use this when:

- Your branch adds new functionality
- Your changes are the primary goal of this rebase
- Upstream changes are mostly refactoring/formatting
- Upstream's changes architecturally conflict with your custom work

**Strategy B: Start from upstream (when upstream changes are larger)**

```bash
git checkout --ours <file>      # Start from UPSTREAM version
# Then manually apply your changes on top
# Read your diff: git diff :1:<file> :3:<file>
```

Use this only when:

- Upstream made significant structural changes
- Your changes are minimal additions that can be easily re-applied
- **You MUST then manually re-apply ALL your changes on top**

**Strategy C: Manual edit (when both sides have significant, interleaved changes)**

Read the file with conflict markers and resolve each `<<<<<<<` block individually, keeping the correct code from both sides. **Ensure every change from YOUR branch is preserved.**

#### 3e. Verify no conflict markers remain

```bash
grep -c '<<<<<<' <file>   # Must return 0
grep -c '>>>>>>>' <file>  # Must return 0
```

### Step 4: Stage and Continue

```bash
git add .
GIT_EDITOR=true git rebase --continue
```

If more conflicts appear in subsequent commits, repeat from Step 1.

## Post-Resolution Verification

**This step is MANDATORY after every rebase with conflicts. DO NOT PUSH until verification passes.**

After the rebase completes, verify that your branch's key additions were NOT silently dropped:

### 0. Verify all branch changes are preserved (CRITICAL)

Your branch's changes MUST still be present. Compare against the patch you saved earlier:

```bash
# Compare current state against your saved branch changes
diff -u /tmp/my-branch-changes.patch <(git diff $DEFAULT...HEAD)

# If the diff shows significant removals, your changes were dropped!
# Alternatively, verify the key changes from your branch:
git diff $DEFAULT...HEAD --stat   # Should show similar files as before rebase
```

**If your branch's changes are missing or significantly different, STOP and reset:**

```bash
git reflog --oneline -10    # Find pre-rebase hash
git reset --hard <pre-rebase-hash>
git rebase $DEFAULT              # Redo with proper Smart Analysis
```

### 1. Diff against the pre-rebase commit

```bash
# Find the pre-rebase commit from reflog
git reflog --oneline -10

# Compare current state vs pre-rebase to find dropped changes
# The pre-rebase commit hash is the one before "rebase (start)"
git diff <pre-rebase-hash> HEAD -- <files-you-changed>
```

If this diff shows YOUR additions missing (code present in pre-rebase but not in current HEAD), those changes were dropped during conflict resolution. You must fix this.

### 2. Verify key symbols exist

For each source file that had conflicts, grep for key functions/exports your branch added:

```bash
# Example: verify your branch's additions still exist
grep -c 'yourFunctionName' <file>
grep -c 'yourExportName' <file>
```

If any return 0, your changes were dropped. Reset and redo:

```bash
git reset --hard <pre-rebase-hash>
git rebase $DEFAULT
# This time, resolve conflicts correctly using Smart Analysis
```

### 3. Check for typecheck/compile errors from missing symbols

```bash
# Run typecheck to catch missing references
# If errors reference functions YOUR branch added, conflict resolution was wrong
```

## Escape Hatches

If resolution becomes too complex:

```bash
# Abort and start over
git rebase --abort

# Skip this specific commit (loses your changes in that commit!)
git rebase --skip

# Save your work and get help
git diff > /tmp/my-resolution-attempt.patch
git rebase --abort
```

## Pre-Push Checklist

Before pushing, verify ALL of the following:

- [ ] Smart Analysis was used for ALL source code conflicts (no blind `--ours`/`--theirs`)
- [ ] All branch changes are preserved (compare `git diff $DEFAULT...HEAD` vs saved patch)
- [ ] No conflict markers remain in any file (`grep -r '<<<<<<' .`)
- [ ] Key functions/exports from your branch still exist
- [ ] Typecheck passes without errors referencing YOUR additions

**If ANY check fails, DO NOT PUSH. Reset and redo the rebase with proper conflict resolution.**

## Quick checks

```bash
git status
git branch --show-current
git rev-parse --abbrev-ref @{u}
```
