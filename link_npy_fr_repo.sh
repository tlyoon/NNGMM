#! /bin/bash
cwd=$(pwd)
hnint=$(hostname | awk '{print substr($0, length($0)-1)}')

#wd=$HOME'/dakota/dakota-gmt/data_generation/v3/record_small/data_train/data_repo'
wd='data_train'
mkdir -p $wd/'data_repo'
mkdir -p $wd/'data_used'
if [ ! -e 'data_test' ]; then
	ln -s $HOME'/dakota/dakota-gmt/data_generation/data_test' .
fi

repo_bk=$HOME'/dakota/dakota-gmt/data_generation/data_repo_bk'
cd $repo_bk
dirs=$(ls -d *)
if [ -z "$dirs" ]; then
   dirs=p1@c$hnint
fi

cd $cwd/$wd/'data_repo'
for i in $dirs:
	do
		i=$(echo $i | awk -F":" '{print $1}')
		f1=$(ls  $repo_bk/$i | grep '_c_' | awk 'NR==1 {print $1}')
		f2=$(ls  $repo_bk/$i | grep '_c_' | awk 'NR==2 {print $1}')

		if [ ! -e $f1 ] && [ ! -e '../data_used/'$f1 ]  && [ ! -e '../'$f1 ];
		then
			ln -s $repo_bk/$i/$f1 '.'
		fi

		if [ ! -e $f2 ] && [ ! -e '../data_used/'$f2 ] && [ ! -e '../'$f2 ];
        	then
			ln -s $repo_bk/$i/$f2 '.'
	        fi
	done
