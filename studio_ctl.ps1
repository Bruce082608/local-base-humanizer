param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "restart", "status", "menu")]
    [string]$Action = "menu",

    [int]$Port = 8765,
    [switch]$NoBrowser
)

$ErrorActionPreference = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Url     = "http://127.0.0.1:$Port"
$PidFile = Join-Path $RootDir ".studio.pid"
$LogOut  = Join-Path $RootDir ".studio.log"
$LogErr  = Join-Path $RootDir ".studio.log.err"

function Get-StudioPid {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn -and $conn.OwningProcess -gt 0) {
        return [int]$conn.OwningProcess
    }
    if (Test-Path $PidFile) {
        $raw = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($raw -and $raw.Trim() -match '^\d+$') {
            $p = Get-Process -Id ([int]$raw.Trim()) -ErrorAction SilentlyContinue
            if ($p) { return [int]$p.Id }
        }
    }
    return $null
}

function Start-Studio {
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "  Local Base-Model Humanizer Studio - 一键启动" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan

    $existingPid = Get-StudioPid
    if ($existingPid) {
        Write-Host "[提示] 工作台已在运行 (PID=$existingPid) -> $Url" -ForegroundColor Yellow
        if (-not $NoBrowser) { Start-Process $Url }
        return 0
    }

    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pyCmd) {
        Write-Host "[错误] 未找到 python，请先安装 Python 3.10+ 并添加到 PATH。" -ForegroundColor Red
        return 1
    }

    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 -UseBasicParsing
        Write-Host "[OK]   检测到本地 Ollama 正在运行 (127.0.0.1:11434)" -ForegroundColor Green
    } catch {
        Write-Host "[提示] 未检测到本地 Ollama (127.0.0.1:11434)，请确认已启动 Ollama。" -ForegroundColor Yellow
    }

    $proc = Start-Process -FilePath $pyCmd.Source `
        -ArgumentList @("app.py", "--no-browser", "--port", "$Port") `
        -WorkingDirectory $RootDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $LogOut `
        -RedirectStandardError $LogErr `
        -PassThru

    Set-Content -Path $PidFile -Value $proc.Id -Encoding Ascii

    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 200
        try {
            $resp = Invoke-WebRequest -Uri "$Url/api/defaults" -TimeoutSec 1 -UseBasicParsing
            if ($resp.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
    }

    if ($ready) {
        Write-Host "[成功] 工作台已后台启动 (PID=$($proc.Id)) -> $Url" -ForegroundColor Green
        Write-Host "       关闭服务请双击同目录下的 stop.bat" -ForegroundColor Gray
        if (-not $NoBrowser) { Start-Process $Url }
        return 0
    } else {
        Write-Host "[错误] 启动超时，请查看 .studio.log.err 排查原因。" -ForegroundColor Red
        return 1
    }
}

function Stop-Studio {
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "  Local Base-Model Humanizer Studio - 一键关闭" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan

    $stopped = New-Object System.Collections.Generic.List[int]

    if (Test-Path $PidFile) {
        $raw = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($raw -and $raw.Trim() -match '^\d+$') {
            $savedPid = [int]$raw.Trim()
            $p = Get-Process -Id $savedPid -ErrorAction SilentlyContinue
            if ($p) {
                Stop-Process -Id $savedPid -Force -ErrorAction SilentlyContinue
                $stopped.Add($savedPid)
            }
        }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        $owner = [int]$c.OwningProcess
        if ($owner -gt 0 -and (-not $stopped.Contains($owner))) {
            Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
            $stopped.Add($owner)
        }
    }

    Start-Sleep -Milliseconds 300

    if ($stopped.Count -gt 0) {
        Write-Host ("[成功] 已关闭工作台服务 (PID: " + ($stopped -join ", ") + ")，端口 $Port 已释放。") -ForegroundColor Green
    } else {
        Write-Host "[提示] 当前未检测到运行中的工作台服务 (端口 $Port 空闲)。" -ForegroundColor Yellow
    }
    return 0
}

function Show-Status {
    $runningPid = Get-StudioPid
    if ($runningPid) {
        Write-Host "  当前状态: [运行中]  PID=$runningPid  地址: $Url" -ForegroundColor Green
    } else {
        Write-Host "  当前状态: [已停止]  端口 $Port 空闲" -ForegroundColor Yellow
    }
}

function Show-Menu {
    while ($true) {
        Clear-Host
        Write-Host "==================================================" -ForegroundColor Cyan
        Write-Host "  Local Base-Model Humanizer Studio 管理控制台" -ForegroundColor Cyan
        Write-Host "==================================================" -ForegroundColor Cyan
        Show-Status
        Write-Host ""
        Write-Host "  [1] 一键启动并打开工作台"
        Write-Host "  [2] 一键关闭工作台"
        Write-Host "  [3] 重启工作台"
        Write-Host "  [4] 仅打开浏览器页面"
        Write-Host "  [0] 退出"
        Write-Host ""
        $choice = Read-Host "请选择操作 [0-4]"
        switch ($choice) {
            "1" { $null = Start-Studio; Start-Sleep -Seconds 2 }
            "2" { $null = Stop-Studio; Start-Sleep -Seconds 2 }
            "3" { $null = Stop-Studio; $null = Start-Studio; Start-Sleep -Seconds 2 }
            "4" { Start-Process $Url }
            "0" { return 0 }
        }
    }
}

switch ($Action) {
    "start"   { exit (Start-Studio) }
    "stop"    { exit (Stop-Studio) }
    "restart" { $null = Stop-Studio; exit (Start-Studio) }
    "status"  { Show-Status; exit 0 }
    "menu"    { exit (Show-Menu) }
}
