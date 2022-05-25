#!/bin/bash
set -o errexit

# Requirements: bash, dash, ksh, yash, zsh, bmake and make

artifactsdir=artifacts/artifacts-test
make_array=( bmake make )
shell_array=( bash dash ksh yash zsh )
for make_element in "${make_array[@]}"
do
	for shell_element in "${shell_array[@]}"
	do
		mkdir -p ${artifactsdir}/python/
		cp python/Makefile ${artifactsdir}/python/Makefile
		"$make_element" -C ${artifactsdir}/python/ SHELL="$shell_element"
		"$make_element" -C ${artifactsdir}/python/ SHELL="$shell_element" clean
		cp Makefile ${artifactsdir}/Makefile
		"$make_element" -C ${artifactsdir}/ SHELL="$shell_element"
		"$make_element" -C ${artifactsdir}/ SHELL="$shell_element" artifacts/tex.tar
		"$make_element" -C ${artifactsdir}/ SHELL="$shell_element" artifacts/tex-lint
		"$make_element" -C ${artifactsdir}/ SHELL="$shell_element" clean
		rm -rf ${artifactsdir}/
	done
done
