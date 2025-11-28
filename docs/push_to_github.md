# Saving Local Changes to GitHub

These steps assume you have a Git remote named `origin` that points to your GitHub repository and that you are on the branch you want to push.

1. **Confirm the remote points at GitHub**
   - List remotes and make sure `origin` is the repository you want:
   ```bash
   git remote -v
   ```
   - If you need to fix it, set the correct URL (SSH or HTTPS):
   ```bash
   git remote set-url origin git@github.com:<your-account>/<repo>.git
   # or
   git remote set-url origin https://github.com/<your-account>/<repo>.git
   ```

2. **Review the current branch and status**
   ```bash
   git status -sb
   git branch --show-current
   ```

2. **Stage only the files you want to publish**
   - Avoid staging generated assets (e.g., `node_modules/`).
   - Use targeted adds to keep commits clean:
   ```bash
   git add path/to/file1 path/to/file2
   ```

3. **Create a descriptive commit**
   ```bash
   git commit -m "<concise summary of the change>"
   ```

4. **Push to GitHub**
   ```bash
   git push origin $(git branch --show-current)
   ```

5. **Authenticate when prompted**
   - SSH: ensure your SSH agent is running and has a loaded key (`ssh-add -l`).
   - HTTPS: use a personal access token when Git prompts for a password.

6. **Verify on GitHub**
   - Open the repository page and confirm the branch shows your new commit.

## Tips
- If you need a new branch instead of pushing to the current one:
  ```bash
  git switch -c feature/my-change
  git push -u origin feature/my-change
  ```
- If `git push` fails because the remote branch moved, fetch and rebase:
  ```bash
 git fetch origin
  git rebase origin/$(git branch --show-current)
  git push origin $(git branch --show-current)
  ```
- Use `git log --oneline --decorate -5` to confirm your commits before pushing.
- Use `git log --oneline --decorate -5` to confirm your commits before pushing.
- To automate the checks above, run the helper script (it will refuse to push if the working tree is dirty or behind the remote):
  ```bash
  ./scripts/push_current_branch.sh            # pushes to origin
  ./scripts/push_current_branch.sh upstream   # pushes to another configured remote
  ```
