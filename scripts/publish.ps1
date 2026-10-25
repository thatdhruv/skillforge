# Creates the GitHub repository and pushes. Run once after `gh auth login`.
#   pwsh scripts/publish.ps1 [-Name skillforge] [-Visibility public]
param([string]$Name = "skillforge", [ValidateSet("public", "private")][string]$Visibility = "public")
$ErrorActionPreference = "Stop"
gh auth status | Out-Null
gh repo create $Name --$Visibility --source . --remote origin --push `
  --description "A self-improving agent with verified skill memory, plus an honest evaluation of whether it improves"
gh repo edit --add-topic langgraph --add-topic mcp --add-topic agents --add-topic evaluation --add-topic pgvector --add-topic kubernetes
