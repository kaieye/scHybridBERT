# "Zeisel" "Klein" "AD-brain" "PBMC-4k" "PBMC-7k" "PBMC-Kang-A" "PBMC-Kang-B" "PBMC-Kang-C"
for i in "Mouse"
do
    python3 produce.py --data_name $i
    # python3 -u finetune.py --data_name $i --cor_embed 0 >> ${i}.log #不使用调控嵌入
    python3 -u GAT_SAGE.py --data_name $i --dis_method 3 --model GraphSAGEGAT >> ${i}.log
    python3 -u finetune.py --data_name $i --cor_embed 1 >> ${i}.log # 使用调控嵌入
    # python3 -u merge.py --data_name $i >> ${i}.log
done
 
