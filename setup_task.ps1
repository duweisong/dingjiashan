# 以管理员身份运行此脚本
 = New-ScheduledTaskAction -Execute "C:\AI\daily_run.bat"
 = New-ScheduledTaskTrigger -Daily -At "17:30"
Register-ScheduledTask -TaskName "DingjiashanV4_Daily" -Action  -Trigger  -Force
Write-Host "[OK] 每天17:30自动运行, 验证: taskschd.msc"
