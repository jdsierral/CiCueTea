function packageToolbox()
%PACKAGETOOLBOX Build the CiCueTea MATLAB toolbox (.mltbx) for File Exchange.
%   Run from anywhere; paths are resolved relative to this file. Version is
%   read from the repo-root VERSION file, so it stays in sync automatically.
%
%   The Identifier below is fixed and must never change between releases —
%   it is how MATLAB's Add-On Manager recognizes different versions of the
%   same toolbox.
%
%   ToolboxFolder is set to Matlab/ (not Matlab/src) so that the packaged
%   file's internal paths are the sane repo-relative ones (src/...,
%   demo/GettingStarted.mlx) rather than an absolute-path mirror of this
%   machine — files outside ToolboxFolder get included at their absolute
%   path. ToolboxFiles and ToolboxMatlabPath are then pruned by hand:
%   only src/** ships and is added to the path; demo/ contributes just the
%   Getting Started guide and its Psychoacoustics/ support functions (not
%   the raw Demo*.m scripts or build output) — bundled so the guide runs,
%   but deliberately off ToolboxMatlabPath, since those conversions are
%   example content, not part of the library's public surface.

repoRoot = fileparts(mfilename('fullpath'));
repoRoot = fileparts(repoRoot);

matlabRoot = fullfile(repoRoot, 'Matlab');
srcRoot = fullfile(matlabRoot, 'src');
gettingStartedGuide = fullfile(matlabRoot, 'demo', 'GettingStarted.mlx');
psychoacousticsDir = fullfile(matlabRoot, 'demo', 'Psychoacoustics');
psychoacousticsFiles = dir(fullfile(psychoacousticsDir, '*.m'));
psychoacousticsFiles = string(fullfile({psychoacousticsFiles.folder}, {psychoacousticsFiles.name}));

versionStr = strtrim(fileread(fullfile(repoRoot, 'VERSION')));

opts = matlab.addons.toolbox.ToolboxOptions( ...
    matlabRoot, "e0fb479d-4f7d-4b6e-aab4-6de6bacbf032");

opts.ToolboxName = "CiCueTea";
opts.ToolboxVersion = versionStr;
opts.AuthorName = "Juan Sierra";
opts.AuthorCompany = "New York University Abu Dhabi";
opts.Summary = "Real-time, invertible Constant-Q Transform (CQT) engine based on nonstationary Gabor frames.";
opts.Description = "CiCueTea is a real-time, invertible Constant-Q Transform (CQT) engine " + ...
    "based on nonstationary Gabor frames (NSGF). This is the MATLAB reference implementation; " + ...
    "the real-time, allocation-free engine and the Python reference implementation live at " + ...
    "https://github.com/jdsierral/CiCueTea.";

srcRootPrefix = string(srcRoot) + filesep;
isUnderSrc = startsWith(opts.ToolboxFiles, srcRootPrefix);
opts.ToolboxFiles = [opts.ToolboxFiles(isUnderSrc); string(gettingStartedGuide); psychoacousticsFiles(:)];
opts.ToolboxMatlabPath = opts.ToolboxMatlabPath(startsWith(opts.ToolboxMatlabPath, string(srcRoot)));
opts.ToolboxGettingStartedGuide = gettingStartedGuide;
opts.OutputFile = fullfile(matlabRoot, 'build', "CiCueTea-" + versionStr + ".mltbx");

matlab.addons.toolbox.packageToolbox(opts);

fprintf("Packaged %s\n", opts.OutputFile);

end
