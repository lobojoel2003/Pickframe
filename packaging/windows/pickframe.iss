; Inno Setup script for the Pickframe Windows installer.
;   iscc /DAppVersion=1.0.0 /DAppBuild=42 /DSourceDir=dist\Pickframe /DOutputName=Pickframe-1.0.0-windows-x64-setup packaging\windows\pickframe.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef AppBuild
  #define AppBuild "0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\Pickframe"
#endif
#ifndef OutputName
  #define OutputName "Pickframe-setup"
#endif

[Setup]
AppId={{7B0C5E36-2F0E-4C51-9D1B-6A4C3E2B8F11}
AppName=Pickframe
AppVersion={#AppVersion}
AppVerName=Pickframe {#AppVersion}
VersionInfoVersion={#AppVersion}.{#AppBuild}
VersionInfoProductVersion={#AppVersion}
AppPublisher=Pickframe
AppPublisherURL=https://github.com/lobojoel2003/Pickframe
AppSupportURL=https://github.com/lobojoel2003/Pickframe/issues
AppUpdatesURL=https://github.com/lobojoel2003/Pickframe/releases
DefaultDirName={autopf}\Pickframe
DefaultGroupName=Pickframe
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist
OutputBaseFilename={#OutputName}
SetupIconFile=..\icons\pickframe.ico
UninstallDisplayIcon={app}\Pickframe.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
ChangesEnvironment=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "Add pickframe-cli to PATH"; GroupDescription: "Command line:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Pickframe"; Filename: "{app}\Pickframe.exe"
Name: "{group}\Uninstall Pickframe"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Pickframe"; Filename: "{app}\Pickframe.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\Pickframe.exe"; Description: "{cm:LaunchProgram,Pickframe}"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Param) + ';', ';' + Uppercase(OrigPath) + ';') = 0;
end;
