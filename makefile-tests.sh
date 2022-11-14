#!/bin/sh
set -o errexit

# Requirements: ash, bash, dash, ksh, posh, yash, zsh, bmake and make

make_array=( bmake make )
shell_array=( ash bash dash ksh posh yash zsh )
for make_element in "${make_array[@]}"
do
	for shell_element in "${shell_array[@]}"
	do
		mkdir -p bin/bin-test/python/
		cp python/Makefile bin/bin-test/python/Makefile
		"$make_element" -C bin/bin-test/python/ SHELL="$shell_element" all || rm -rf bin/bin-test
		"$make_element" -C bin/bin-test/python/ SHELL="$shell_element" check || rm -rf bin/bin-test
		"$make_element" -C bin/bin-test/python/ SHELL="$shell_element" clean || rm -rf bin/bin-test
		cp Makefile bin/bin-test/Makefile
		"$make_element" -C bin/bin-test/ SHELL="$shell_element" all || rm -rf bin/bin-test
		"$make_element" -C bin/bin-test/ SHELL="$shell_element" check || rm -rf bin/bin-test
		"$make_element" -C bin/bin-test/ SHELL="$shell_element" clean || rm -rf bin/bin-test
		rm -rf bin/bin-test/
	done
done
