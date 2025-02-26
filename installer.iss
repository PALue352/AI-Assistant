; AI_Assistant/installer.iss
[Setup]
AppName=AI Assistant
AppVersion=0.1.0
DefaultDirName={userdocs}\AI_Assistant
DefaultGroupName=AI Assistant
OutputDir=dist
OutputBaseFilename=AI_Assistant_Installer
Compression=lzma
SolidCompression=yes
SetupIconFile=C:\Users\User\Desktop\Grok AI\AI_Assistant\icon.ico
WizardStyle=modern
DirExistsWarning=no
UninstallDisplayName=AI Assistant Uninstall
Uninstallable=yes
DiskSpanning=yes
DiskSliceSize=2100000000
SlicesPerDisk=1
UseSetupLdr=yes

[Files]
Source: "requirements.txt"; DestDir: "{app}"
Source: "install_dependencies.bat"; DestDir: "{app}"
Source: "launch_ai.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "ai_assistant/*"; DestDir: "{app}/ai_assistant"; Flags: recursesubdirs createallsubdirs
Source: "ai_assistant/__init__.py"; DestDir: "{app}/ai_assistant"; Flags: onlyifdoesntexist
Source: ".env"; DestDir: "{app}"
Source: "python-3.12.8-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "build.bat"; DestDir: "{app}"
Source: "icon.ico"; DestDir: "{app}"
Source: "plugins/__init__.py"; DestDir: "{app}/plugins"; Flags: onlyifdoesntexist
Source: "ai_assistant/core/plugin.py"; DestDir: "{app}/ai_assistant/core"; Flags: ignoreversion
Source: "ai_assistant/core/plugin_manager.py"; DestDir: "{app}/ai_assistant/core"; Flags: ignoreversion
Source: "download_models.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "install_pytorch.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "ai_assistant/core/training_data/sub_ai_usage.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/ai_trainer_instructions.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/advisor_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/hardware_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/sales_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/motion_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/av_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/decoder_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/analyzer_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/business_mgr_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/marketing_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/spatial_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/ethical_training_data.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/medical_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/knowledge_base.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/patent_templates.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/cognitive_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/financial_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/pr_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/common_sense_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"
Source: "ai_assistant/core/training_data/protection_kb.json"; DestDir: "{app}/ai_assistant/core/training_data"

[Dirs]
Name: "{app}/plugins"; Flags: uninsalwayscreate
Name: "{app}/AI_Assistant"; Flags: uninsalwayscreate
Name: "{app}/Resources"; Flags: uninsalwayscreate
Name: "{app}/model_cache"; Flags: uninsalwayscreate

[Icons]
Name: "{userdesktop}\AI Assistant"; Filename: "{app}\launch_ai.bat"; WorkingDir: "{app}"; IconFilename: "{app}/icon.ico"; Tasks: desktopicon
Name: "{group}\AI Assistant"; Filename: "{app}\launch_ai.bat"; WorkingDir: "{app}"; IconFilename: "{app}/icon.ico"
Name: "{userstartup}\AI Assistant"; Filename: "{app}\launch_ai.bat"; WorkingDir: "{app}"; IconFilename: "{app}/icon.ico"; Tasks: startupicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: checkedonce
Name: "startupicon"; Description: "Launch AI Assistant at &startup"; GroupDescription: "Additional icons:"
Name: "gpu_install"; Description: "Install GPU version of PyTorch (requires NVIDIA GPU)"; GroupDescription: "PyTorch Options:"; Flags: unchecked

[Run]
Filename: "{tmp}\python-3.12.8-amd64.exe"; Parameters: "/quiet InstallAllUsers=0 PrependPath=1 TargetDir=""{code:GetPythonDir}"""; Check: PythonNotInstalled; Flags: waituntilterminated
Filename: "{app}\install_dependencies.bat"; Parameters: ""; Description: "Install Dependencies"; Flags: runasoriginaluser waituntilterminated
Filename: "{app}\install_pytorch.bat"; Parameters: "{code:GetPyTorchParams}"; Description: "Install PyTorch"; Flags: runasoriginaluser waituntilterminated
Filename: "{app}\download_models.py"; Parameters: ""; Description: "Download Pre-Loaded Models"; Flags: runasoriginaluser waituntilterminated

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
var
  PythonDirPage: TInputDirWizardPage;

function CheckCppBuildTools: Boolean;
var
  ResultCode: Integer;
begin
  if not Exec(ExpandConstant('{cmd}'), '/c where cl >nul 2>&1', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    Result := MsgBox('Microsoft C++ Build Tools are required for some dependencies. Would you like to install them now?', mbConfirmation, MB_YESNO) = IDYES;
    if Result then
    begin
      if not Exec(ExpandConstant('{cmd}'), '/c start https://visualstudio.microsoft.com/visual-cpp-build-tools/', '', SW_SHOW, ewNoWait, ResultCode) then
      begin
        MsgBox('Failed to open C++ Build Tools download page. Please install manually from https://visualstudio.microsoft.com/visual-cpp-build-tools/', mbError, MB_OK);
        Result := False;
        Exit;
      end;
      MsgBox('Please install C++ Build Tools, restart your computer, and rerun the installer.', mbInformation, MB_OK);
      Result := False;
    end else
    begin
      MsgBox('Installation cannot continue without C++ Build Tools. Please install manually from https://visualstudio.microsoft.com/visual-cpp-build-tools/ and retry.', mbError, MB_OK);
      Result := False;
    end;
  end else
    Result := True;
end;

function PythonNotInstalled: Boolean;
begin
  Result := True;  ; Assume Python isn’t installed—installer will handle it
end;

procedure InitializeWizard;
begin
  if not CheckCppBuildTools then
    Abort;
  PythonDirPage := CreateInputDirPage(wpSelectDir,
    'Select Python Installation Directory', 'Where should Python be installed?',
    'Select the directory where Python 3.12.8 should be installed. This can be separate from the AI Assistant directory.',
    False, '');
  PythonDirPage.Add('');
  PythonDirPage.Values[0] := ExpandConstant('{userdocs}\Python312');
end;

function GetPythonDir(Param: String): String;
begin
  Result := PythonDirPage.Values[0];
end;

function GetPyTorchParams(Param: String): String;
begin
  if IsTaskSelected('gpu_install') then
    Result := 'gpu'
  else
    Result := 'cpu';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    if not Exec(ExpandConstant('{cmd}'), '/c icacls "{app}" /grant Users:F /T', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      Log('Failed to set permissions on {app}');
  end;
end;