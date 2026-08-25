"""
tools/push_to_github.py - Maintainer-only helper to push this project to its GitHub
remote via a personal access token. NOT part of the RFM-T platform itself (app.py,
src/*, generate_action_plan.py) and not required to install, run, or test the
project — see the Installation section of README.md for the actual setup steps.
"""

import sys
import os
from dulwich.repo import Repo
import dulwich.porcelain as git

def push_to_remote(token=None):
    repo = Repo('.')
    remote_url = "https://github.com/shambo1597-afk/rfm-customer-intelligence-engine"
    
    if not token:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]
        
    if token:
        # Push using authenticated token URL
        auth_url = f"https://{token}@github.com/shambo1597-afk/rfm-customer-intelligence-engine.git"
        print(f"[*] Pushing with token authentication to {remote_url} on branch 'main'...")
        try:
            git.push(repo, auth_url, 'refs/heads/main:refs/heads/main')
            print("✅ Push succeeded! Repository is live on GitHub.")
            return True
        except Exception as e:
            print(f"❌ Push error: {e}")
            return False
    else:
        print("[!] No GitHub Token provided.")
        print("    Usage: python push_to_github.py <YOUR_GITHUB_PERSONAL_ACCESS_TOKEN>")
        print("    Or set the GITHUB_TOKEN environment variable.")
        return False

if __name__ == "__main__":
    push_to_remote()
