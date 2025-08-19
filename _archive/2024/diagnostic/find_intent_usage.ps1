# PowerShell command to find $intentTracker usage:
cd C:\Users\jason\Desktop\tori\kha\tori_ui_svelte\src\routes
Select-String -Path "+page.svelte" -Pattern '\$intentTracker' -Context 2,2

# This will show where $intentTracker is used with 2 lines before and after