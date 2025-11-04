# GitHub Upload Instructions

## Method 1: Using Git Command Line (Recommended)

### Step 1: Initialize Git Repository
```bash
cd Financial-Analytics-Derivatives
git init
```

### Step 2: Add All Files
```bash
git add .
git status  # Verify files are staged
```

### Step 3: Create Initial Commit
```bash
git commit -m "Initial commit: Financial Analytics & Derivatives Project

- Portfolio optimization with Efficient Frontier
- Monte Carlo simulation for risk analysis
- Black-Scholes option pricing model
- Put-Call Parity arbitrage detection
- Comprehensive visualizations (9+ plots)"
```

### Step 4: Add Remote Repository
```bash
git remote add origin https://github.com/Kluxuva/Financial-Analytics-Derivatives.git
```

### Step 5: Push to GitHub
```bash
# First time push
git push -u origin main

# OR if using master branch
git branch -M main
git push -u origin main
```

### Step 6: Verify Upload
Visit: https://github.com/Kluxuva/Financial-Analytics-Derivatives

---

## Method 2: Using GitHub Desktop

### Step 1: Open GitHub Desktop
- Download from: https://desktop.github.com/

### Step 2: Add Repository
1. Click "File" → "Add Local Repository"
2. Navigate to `Financial-Analytics-Derivatives` folder
3. Click "Add Repository"

### Step 3: Create Repository
1. Click "Publish repository"
2. Name: Financial-Analytics-Derivatives
3. Description: Portfolio optimization and option pricing analysis
4. Uncheck "Keep this code private" (or check if you want private)
5. Click "Publish Repository"

---

## Method 3: Using GitHub Web Interface

### Step 1: Create ZIP File
```bash
cd ..
zip -r Financial-Analytics-Derivatives.zip Financial-Analytics-Derivatives/ -x "*.git*" "**/__pycache__/*"
```

### Step 2: Upload via GitHub
1. Go to https://github.com/Kluxuva/Financial-Analytics-Derivatives
2. Click "uploading an existing file"
3. Drag and drop files or select them
4. Add commit message
5. Click "Commit changes"

---

## Troubleshooting

### Issue: Repository already exists

**Solution:**
```bash
# Remove existing .git folder
rm -rf .git

# Start fresh
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/Kluxuva/Financial-Analytics-Derivatives.git
git push -u origin main --force
```

### Issue: Authentication failed

**Solution 1 - Use Personal Access Token (PAT):**
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic) with 'repo' scope
3. Use token as password when pushing

**Solution 2 - Use SSH:**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub
# Copy public key
cat ~/.ssh/id_ed25519.pub

# Go to GitHub → Settings → SSH and GPG keys → New SSH key
# Paste the key

# Change remote to SSH
git remote set-url origin git@github.com:Kluxuva/Financial-Analytics-Derivatives.git
git push -u origin main
```

### Issue: Large files rejected

**Solution:**
```bash
# Remove plots from tracking (they'll be generated locally)
echo "plots/*.png" >> .gitignore
git rm --cached plots/*.png
git commit -m "Remove large plot files from tracking"
git push
```

---

## Post-Upload Checklist

After successfully uploading:

- [ ] Verify all files are visible on GitHub
- [ ] Check README.md renders correctly
- [ ] Ensure .gitignore is working (no __pycache__ or .pyc files)
- [ ] Add topics/tags: `python`, `finance`, `portfolio-optimization`, `black-scholes`, `monte-carlo`
- [ ] Add repository description
- [ ] Enable GitHub Pages (optional) for documentation
- [ ] Add LICENSE file
- [ ] Star your own repository!

---

## Adding to Existing Repository

If the repository already has content:

```bash
cd Financial-Analytics-Derivatives
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/Kluxuva/Financial-Analytics-Derivatives.git
git fetch origin
git merge origin/main --allow-unrelated-histories
git push -u origin main
```

---

## Best Practices

### 1. Update .gitignore
Already included, but verify:
```
__pycache__/
*.pyc
.venv/
plots/*.png
```

### 2. Use Meaningful Commit Messages
```bash
# Good
git commit -m "Add Monte Carlo simulation with 1000 paths"
git commit -m "Fix: Option pricing calculation for out-of-money options"
git commit -m "Docs: Update README with installation instructions"

# Bad
git commit -m "update"
git commit -m "fix stuff"
git commit -m "changes"
```

### 3. Commit Often
```bash
# After each feature
git add financial_analysis.py
git commit -m "Add: Efficient Frontier visualization"

git add README.md
git commit -m "Docs: Add Black-Scholes explanation"

# Push when ready
git push
```

### 4. Check Status Before Committing
```bash
git status           # See what's changed
git diff            # See exact changes
git diff --staged   # See staged changes
```

---

## Useful Git Commands

### Viewing History
```bash
git log                    # View commit history
git log --oneline         # Compact history
git log --graph --all     # Visual branch history
```

### Undoing Changes
```bash
git checkout -- file.py   # Discard changes to file
git reset HEAD file.py    # Unstage file
git revert <commit-hash>  # Undo a commit
```

### Branching
```bash
git branch feature-name           # Create branch
git checkout feature-name         # Switch branch
git checkout -b feature-name      # Create and switch

git merge feature-name            # Merge branch
git branch -d feature-name        # Delete branch
```

---

## Advanced: Automating Updates

### Create Update Script
```bash
# update_github.sh
#!/bin/bash

git add .
git commit -m "Auto-update: $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main
echo "✓ Successfully pushed to GitHub"
```

```bash
chmod +x update_github.sh
./update_github.sh
```

---

## Repository Settings

After upload, configure these settings on GitHub:

### 1. About Section
- Description: "Portfolio optimization using Efficient Frontier & Monte Carlo + Option pricing with Black-Scholes"
- Website: (optional)
- Topics: `python`, `finance`, `portfolio-optimization`, `derivatives`, `quantitative-finance`

### 2. Features
- ✓ Issues
- ✓ Projects
- ✓ Wiki (optional)
- ✓ Discussions (optional)

### 3. Social Preview
Upload a preview image (one of your plots):
- Settings → Options → Social preview → Upload image

### 4. Branch Protection (Optional)
For collaboration:
- Settings → Branches → Add rule
- Require pull request reviews
- Require status checks

---

## Quick Reference

```bash
# Initial setup
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/Kluxuva/Financial-Analytics-Derivatives.git
git push -u origin main

# Regular updates
git add .
git commit -m "Your message"
git push

# Pull latest changes
git pull origin main

# Clone repository
git clone https://github.com/Kluxuva/Financial-Analytics-Derivatives.git
```

---

## Need Help?

- **Git Documentation:** https://git-scm.com/doc
- **GitHub Guides:** https://guides.github.com/
- **Git Cheat Sheet:** https://education.github.com/git-cheat-sheet-education.pdf

**Common Error Messages:**
- "fatal: not a git repository" → Run `git init`
- "fatal: remote origin already exists" → Run `git remote remove origin` first
- "Updates were rejected" → Run `git pull origin main` first

---

Ready to share your project with the world! 🚀
