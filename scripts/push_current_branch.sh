#!/usr/bin/env bash
set -euo pipefail

# Push the current branch to the specified remote (default: origin).
# Usage: ./scripts/push_current_branch.sh [remote]

remote=${1:-origin}

# Ensure the repository is clean before pushing.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "✖ Working tree is dirty. Please commit or stash changes before pushing." >&2
  exit 1
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)

echo "Remote: ${remote}" ; echo "Branch: ${current_branch}"

# Validate the remote exists.
if ! git remote get-url "${remote}" >/dev/null 2>&1; then
  echo "✖ Remote '${remote}' is not configured. Add it with:\n   git remote add ${remote} <url>" >&2
  exit 1
fi

# Fetch to ensure we have the latest refs.
git fetch "${remote}" --quiet

# Ensure the local branch is up to date with the remote if it exists.
if git show-ref --quiet "refs/remotes/${remote}/${current_branch}"; then
  if ! git merge-base --is-ancestor "${remote}/${current_branch}" "${current_branch}"; then
    echo "✖ Local branch is behind ${remote}/${current_branch}. Please pull or rebase before pushing." >&2
    exit 1
  fi
fi

echo "Pushing ${current_branch} to ${remote}..."
git push "${remote}" "${current_branch}"
