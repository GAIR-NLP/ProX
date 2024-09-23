prefix="https://data.together.xyz/redpajama-data-v2/v1.0.0/documents/"
suffix=".json.gz"

files=("redpj_sample_documents_urls.txt")
download_dir=$RAW_DATA_DIR/redpj_400B
concurrency=20

for file in "${files[@]}"; do
    cat "$file" | while read -r line; do
        filename=$(echo "$line" | tr '/' '_')
        url="${prefix}${line}${suffix}"
        echo "$url -o $download_dir/$filename$suffix"
    done | xargs -n 3 -P 5 -I {} bash -c 'curl -L {}'
done
