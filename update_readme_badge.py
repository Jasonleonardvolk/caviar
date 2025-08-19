import os
import re
import sys

def update_readme_badge():
    """Update README.md with actual GitHub username/repo"""
    
    # Get GitHub remote URL
    import subprocess
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True
        )
        remote_url = result.stdout.strip()
        
        # Extract username and repo from URL
        # Handles both HTTPS and SSH URLs
        if "github.com" in remote_url:
            if remote_url.startswith("https://"):
                # https://github.com/username/repo.git
                match = re.search(r'github\.com/([^/]+)/([^/\.]+)', remote_url)
            else:
                # git@github.com:username/repo.git
                match = re.search(r'github\.com:([^/]+)/([^/\.]+)', remote_url)
            
            if match:
                username = match.group(1)
                repo = match.group(2)
                print(f"Detected GitHub: {username}/{repo}")
            else:
                print("Could not parse GitHub URL")
                username = input("Enter your GitHub username: ")
                repo = input("Enter your repository name: ")
        else:
            print("Not a GitHub repository")
            username = input("Enter your GitHub username: ")
            repo = input("Enter your repository name: ")
    except:
        print("Git not configured")
        username = input("Enter your GitHub username: ")
        repo = input("Enter your repository name: ")
    
    # Read README
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        print(f"ERROR: {readme_path} not found!")
        return False
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace USERNAME/REPO with actual values
    original = content
    content = content.replace('USERNAME/REPO', f'{username}/{repo}')
    
    if content != original:
        # Write updated README
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated README.md with {username}/{repo}")
        return True
    else:
        print("ℹ️  No USERNAME/REPO placeholders found in README.md")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("README CI Badge Updater")
    print("=" * 50)
    
    if update_readme_badge():
        print("\n✅ README.md updated successfully!")
        print("\nThe CI badge now points to your repository.")
    else:
        print("\n⚠️  Please manually update the CI badge in README.md")
