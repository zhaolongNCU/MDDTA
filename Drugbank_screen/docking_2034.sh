
config_folder="/home/zhaol409100220027/zhaol409100220027/Drugbank_screen/config"

total_files=$(ls "$config_folder"/*.txt | wc -l)

if [ $total_files -eq 0 ]; then
    echo "No .txt files were found in the $config_folder"
    exit 1
fi

current_file=0

for config_file in "$config_folder"/*.txt; do
    if [ -f "$config_file" ]; then
        current_file=$((current_file + 1))
        progress=$((current_file * 100 / total_files))
        echo -n "Processing $current_file/$total_files ($progress%) : "
        seq -s '=' $progress | tr -d '[:digit:]'  
        echo " finished! "
        qvinaw --config "$config_file"
    fi
done

echo "All samples processed!"
