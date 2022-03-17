#!  /bin/bash


files=$(ls -lhrt | awk 'NR>=101' | grep npy | awk 'BEGIN { FS = "/" } ; { print $9}')
for i in $files
	do
         mv $i ../data_repo
	echo mv $i ../data_repo
	done
