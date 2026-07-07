Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\62770\.qoderworkcn\workspace\mql6v86x74ovudgw\outputs\THU-BDC2026\code"
WshShell.Run "cmd /c python -u run_c4_retrain_predict.py > c4_retrain_stdout.log 2>&1", 0, False
