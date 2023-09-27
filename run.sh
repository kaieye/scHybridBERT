for i in   "Mouse" "Zeisel" "Klein" "AD-brain" "PBMC-4k" "PBMC-7k" "PBMC-Kang-A" "PBMC-Kang-B" "PBMC-Kang-C"
do
    python3 -u preprocess.py --data_name $i
    python3 -u Gene_L.py --data_name $i --regu_embed 1  >> ${i}.log 2 >&1
    python3 -u Cell_L.py --data_name $i --dis_method 3  >> ${i}.log 2 >&1
    python3 -u fusion.py --data_name $i>> ${i}.log 2 >&1
done
