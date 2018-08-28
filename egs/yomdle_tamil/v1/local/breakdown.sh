#!/bin/bash

# Authors: Desh Raj

# This script breaks down the test data according to several split options. The available options currently are: line height.
# Usage: ./breakdown.sh -d data_dir [-f input_file.txt] [-s save_dir] [-o height] [-h 30]
# The arguments within [] are optional. Default values are as follows:
# -f NONE -> Takes all files in the data_dir
# -s $PWD -> current directory
# -o height
# -h 30 -> only used if argument -o has value height

data_dir=''
input_file=''
save_dir=''
option=''
height=''

create_file () {
	echo "creating output file $1"
	[ -e $1 ] && rm $1
	touch $1
}

while getopts ":d:f:s:o:h:" opt; do
	case $opt in
		d ) data_dir=$OPTARG;;
		f ) input_file=$OPTARG;;
		s ) save_dir=$OPTARG;;
		o ) option=$OPTARG;;
		h ) height=$OPTARG;;
	esac
done

if [ -z "$data_dir" ]; then
	echo "Missing path to source directory"
	exit 1
elif [ ! -d "$data_dir" ]; then
	echo "Source directory does not exist"
	exit 1
fi

out_file1=''
out_file2=''

if [ "$option" = "height" ] || [ -z "$option" ]; then
	option="height"
	if [ -z "$height" ]; then
		height=30
	fi
	out_file1='test_above'$height
	out_file2='test_below'$height
else
	echo "invalid value for argument -o"
	exit 1
fi

create_file $save_dir$out_file1
create_file $save_dir$out_file2

images=()

if [ -z "$input_file" ]; then 
	for file in "$data_dir"/*; do
		images+=($file)
	done
else
	files=$(cat $input_file |tr "\n" " ")
	images=($files)
	images=("${images[@]/%/.png}")
fi

if [ "$option" = "height" ]; then
	echo "splitting based on $option $height"	
	echo "reading ${#images[@]} files from $data_dir"
	for file in "${images[@]}"; do
		true_height=$(identify -format "%h" "$data_dir""/""$file")	
		if [ $true_height -gt $height ]; then
			echo "$file" >> $save_dir$out_file1
		else
			echo "$file" >> $save_dir$out_file2
		fi
	done
fi


