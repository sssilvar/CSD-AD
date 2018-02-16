import os

# Define parameters
soft_dir = '/home/sssilvar/.local/inria/'
subjects_dir = '/home/sssilvar/FreeSurferSD/'
workspace_dir = '/home/sssilvar/workspace_inria/'

# Normalise paths and add environment variable
soft_dir = os.path.normpath(soft_dir)
subjects_dir = os.path.normpath(subjects_dir)
workspace_dir = os.path.normpath(workspace_dir)

print('[  OK  ] Setting $SUBJECTS_DIR to: ', subjects_dir)
# os.system('export SUBJECTS_DIR=%s' % subjects_dir)

# Set the pipeline commands
hemis = ['lh', 'rh']
pipeline_commands = {}
for hemi in hemis:
    # define subject and workspace subject directory: sd_surf, wd
    sd_surf = os.path.join(subjects_dir, 'subject_id', 'surf', hemi + '.white')
    wd = os.path.join(workspace_dir, 'subject_id')

    pipeline_commands[hemi] = [
        'mris_convert ' + sd_surf + ' ' + os.path.join(wd, hemi + '.white.vtk')
    ]

for key, val in pipeline_commands.items():
    a = str(val).replace('subject_id', 'Pepe')
    print(a)
