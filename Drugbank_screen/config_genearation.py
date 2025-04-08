import os

ligand_folder = '/home/zhaol409100220027/zhaol409100220027/Drugbank_screen/ligand'
output_folder = '/home/zhaol409100220027/zhaol409100220027/Drugbank_screen/config'

config_template = '''receptor = /home/zhaol409100220027/zhaol409100220027/Drugbank_screen/7BQY.pdbqt
ligand = /home/zhaol409100220027/zhaol409100220027/Drugbank_screen/ligand/{id}.pdbqt
center_x = 8.066
center_y = -1.463
center_z = 25.21
size_x = 23.8333333333
size_y = 23.8333333333
size_z = 23.8333333333
out = /home/zhaol409100220027/zhaol409100220027/Drugbank_screen/output/{id}_out_qvinaw.pdbqt
cpu = 12
exhaustiveness = 32
num_modes = 30
energy_range = 4'''

for filename in os.listdir(ligand_folder):
    if filename.endswith('.pdbqt'):
        file_id = os.path.splitext(filename)[0]
        
        config_content = config_template.format(id=file_id)
        
        config_file_path = os.path.join(output_folder, f'{file_id}_config.txt')
        
        with open(config_file_path, 'w') as config_file:
            config_file.write(config_content)
        
        print(f'Config file for {file_id} created: {config_file_path}')
