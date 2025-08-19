$ErrorActionPreference = "Stop"

Write-Host "FIXING COGNITIVE_INTERFACE & CONCEPT_MESH" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$TORI = "C:\Users\jason\Desktop\tori\kha"
$PIGPEN = "C:\Users\jason\Desktop\pigpen"

# 1. Ensure __init__.py files exist in both locations
Write-Host "`nCreating __init__.py files..." -ForegroundColor Yellow

@("$TORI\ingest_pdf", "$PIGPEN\ingest_pdf") | ForEach-Object {
    if (Test-Path $_) {
        $initFile = "$_\__init__.py"
        if (!(Test-Path $initFile)) {
            New-Item -ItemType File -Path $initFile -Force | Out-Null
            Write-Host "  Created: $initFile" -ForegroundColor Green
        }
    }
}

# 2. Create concept_mesh mock in both locations
Write-Host "`nSetting up concept_mesh..." -ForegroundColor Yellow

$conceptMeshCode = @'
"""Mock ConceptMeshConnector"""
class ConceptMeshConnector:
    def __init__(self, url=None):
        self.url = url or "http://localhost:8003/api/mesh"
    def connect(self):
        return True
    def get_concepts(self):
        return []
    def add_concept(self, concept):
        return {"id": "mock_id", "concept": concept}
'@

@("$TORI\concept_mesh", "$PIGPEN\concept_mesh") | ForEach-Object {
    if (!(Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
    }
    Set-Content -Path "$_\__init__.py" -Value $conceptMeshCode
    Write-Host "  Created concept_mesh in: $_" -ForegroundColor Green
}

# 3. Set environment for both
Write-Host "`nEnvironment setup commands:" -ForegroundColor Yellow
Write-Host '  $env:PYTHONPATH = "C:\Users\jason\Desktop\tori\kha"' -ForegroundColor White
Write-Host '  # or for pigpen:' -ForegroundColor Gray
Write-Host '  $env:PYTHONPATH = "C:\Users\jason\Desktop\pigpen"' -ForegroundColor White

Write-Host "`nDONE! Now you can import:" -ForegroundColor Green
Write-Host "  from ingest_pdf.cognitive_interface import add_concept_diff" -ForegroundColor White
Write-Host "  from concept_mesh import ConceptMeshConnector" -ForegroundColor White
