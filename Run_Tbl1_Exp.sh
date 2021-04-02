Start=1
dim=2
num_run=20
num_samples=20000

log_path='./Models/Random_Cov/'
log_path+=$dim
log_path+='d/Samples_'
log_path+=$((num_samples/1000))
log_path+='.0k/Run_'

for i in $(eval echo "{$Start..$num_run}")
do	
	mkdir -p $log_path$i/
	if [ $i -le 10 ]
	then
		python -u Random_Cov.py --Num_Samples $num_samples --GPU 0 --Run $i --Dim $dim >> $log_path$i/output.log &
	else
		python -u Random_Cov.py --Num_Samples $num_samples --GPU 1 --Run $i --Dim $dim >> $log_path$i/output.log &
	fi
done