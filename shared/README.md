# Shared Resources

Drop PDFs, articles, and conversations here for team review.

## How to Share:
1. Add files to this `shared/` folder
2. Run `.\p.ps1` to push instantly
3. Your friend pulls and sees everything

## Subfolders:
- `shared/pdfs/` - Research papers, docs
- `shared/conversations/` - Saved chats
- `shared/articles/` - Web articles (save as .md or .txt)
- `shared/shaders/` - WGSL shader snippets

## Quick Share Commands:
```powershell
# Copy PDF here and push
Copy-Item "C:\Downloads\paper.pdf" "D:\Dev\kha\shared\pdfs\"
.\p.ps1 "added paper"

# Save conversation as markdown
"Conversation content here..." | Out-File "shared\conversations\shader-discussion.md"
.\p.ps1 "shader discussion"
```
