#!/bin/bash 

#Powderday cluster setup convenience script for SLURM queue mananger
#on HiPerGator at the University of FLorida.  This sets up the model
#files for a cosmological simulation where we want to model many
#galaxies at once.

#Notes of interest:

#1. This does *not* set up the parameters_master.py file: it is
#assumed that you will *very carefully* set this up yourself.

#2. This requires bash versions >= 3.0.  To check, type at the shell
#prompt: 

#> echo $BASH_VERSION

n_nodes=$1
model_dir=$2
hydro_dir=$3
model_run_name=$4
COSMOFLAG=$5
FILTERFLAG=$6
model_dir_remote=$7
hydro_dir_remote=$8
xpos=$9
ypos=${10}
zpos=${11}
galaxy=${12}
snap=${13}
tcmb=${14}

echo "processing model file for galaxy,snapshot:  $galaxy,$snap"


#clear the pyc files
rm -f *.pyc

#set up the model_**.py file
echo "setting up the output directory in case it doesnt already exist"
echo "snap is: $snap"
echo "model dir is: $model_dir"
mkdir $model_dir

filem="$model_dir/snap${snap}_galaxy${galaxy}.py"
echo "writing to $filem"
rm -f $filem


echo "#Snapshot Parameters" >> $filem
echo "#<Parameter File Auto-Generated by setup_all_cluster.sh>" >> $filem
echo "snapshot_num =  $snap" >> $filem
echo "galaxy_num = $galaxy" >>$filem
echo -e "\n" >> $filem

echo -e "galaxy_num_str = str(galaxy_num)" >> $filem

#echo "if galaxy_num < 10:" >> $filem
#echo -e "\t galaxy_num_str = '00'+str(galaxy_num)" >> $filem
#echo -e "elif galaxy_num >= 10 and galaxy_num <100:" >> $filem
#echo -e "\t galaxy_num_str = '0'+str(galaxy_num)" >> $filem
#echo -e "else:" >> $filem
#echo -e "\t galaxy_num_str = str(galaxy_num)" >> $filem

echo -e "\n" >>$filem

echo "if snapshot_num < 10:" >> $filem
echo -e "\t snapnum_str = '00'+str(snapshot_num)" >> $filem
echo -e "elif snapshot_num >= 10 and snapshot_num <100:" >> $filem
echo -e "\t snapnum_str = '0'+str(snapshot_num)" >> $filem
echo -e "else:" >> $filem
echo -e "\t snapnum_str = str(snapshot_num)" >> $filem

echo -e "\n" >>$filem

if [ $COSMOFLAG -eq 1 ]
then
    echo "hydro_dir = '$hydro_dir_remote/snapdir_'+snapnum_str+'/'">>$filem
    echo "snapshot_name = 'snapshot_'+snapnum_str+'.0.hdf5'" >>$filem
elif [ $FILTERFLAG -eq 1 ]
then
    echo "hydro_dir = '$hydro_dir_remote/snap'+snapnum_str+'/'">>$filem
    #echo "snapshot_name = 'snap'+snapnum_str+'_galaxy'+galaxy_num_str+'_filtered.hdf5'">>$filem
    echo "snapshot_name = 'galaxy_'+str(galaxy_num)+'.hdf5'">>$filem

else
    echo "hydro_dir = '$hydro_dir_remote/'">>$filem
    echo "snapshot_name = 'snapshot_'+snapnum_str+'.hdf5'" >>$filem
fi


echo -e "\n" >>$filem

echo "#where the files should go" >>$filem
echo "PD_output_dir = '${model_dir_remote}/' ">>$filem
echo "Auto_TF_file = 'snap'+snapnum_str+'.logical' ">>$filem
echo "Auto_dustdens_file = 'snap'+snapnum_str+'.dustdens' ">>$filem

echo -e "\n\n" >>$filem
echo "#===============================================" >>$filem
echo "#FILE I/O" >>$filem
echo "#===============================================" >>$filem
echo "inputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'.rtin'" >>$filem
echo "outputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'.rtout'" >>$filem

echo -e "\n\n" >>$filem
echo "#===============================================" >>$filem
echo "#GRID POSITIONS" >>$filem
echo "#===============================================" >>$filem
echo "x_cent = ${xpos}" >>$filem
echo "y_cent = ${ypos}" >>$filem
echo "z_cent = ${zpos}" >>$filem

echo -e "\n\n" >>$filem
echo "#===============================================" >>$filem
echo "#CMB INFORMATION" >>$filem
echo "#===============================================" >>$filem
echo "TCMB = ${tcmb}" >>$filem

echo "writing slurm submission master script file"
qsubfile="$model_dir/master.snap${snap}.job"
rm -f $qsubfile
echo $qsubfile

echo "#! /bin/bash" >>$qsubfile
echo "#SBATCH --job-name=${model_run_name}.snap${snap}" >>$qsubfile
echo "#SBATCH --output=pd.master.snap${snap}.o" >>$qsubfile
echo "#SBATCH --error=pd.master.snap${snap}.e" >>$qsubfile
echo "#SBATCH --mail-type=ALL" >>$qsubfile
echo "#SBATCH --mail-user=desika.narayanan@gmail.com" >>$qsubfile
echo "#SBATCH --time=48:00:00" >>$qsubfile
echo "#SBATCH --tasks-per-node=32">>$qsubfile
echo "#SBATCH --nodes=$n_nodes">>$qsubfile
echo "#SBATCH --mem-per-cpu=3800">>$qsubfile
echo "#SBATCH --account=narayanan">>$qsubfile
echo "#SBATCH --qos=narayanan-b">>$qsubfile
echo -e "\n">>$qsubfile
echo -e "\n" >>$qsubfile

echo "module purge">>$qsubfile
echo "module load git/1.9.0">>$qsubfile
echo "module load gsl/1.16">>$qsubfile
echo "module load gcc/5.2.0">>$qsubfile
echo "module load hdf5/1.8.16">>$qsubfile
echo "module load openmpi/1.10.2">>$qsubfile
echo -e "\n">>$qsubfile

echo "cd /home/desika.narayanan/pd/">>$qsubfile
echo "python pd_front_end.py $model_dir_remote parameters_master snap${snap}_galaxy\$SLURM_ARRAY_TASK_ID  > $model_dir_remote/snap${snap}_galaxy\$SLURM_ARRAY_TASK_ID.LOG">>$qsubfile


#done
