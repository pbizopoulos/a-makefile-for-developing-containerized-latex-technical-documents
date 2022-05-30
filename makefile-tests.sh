#!/bin/bash
set -o errexit

# Requirements: ash, bash, dash, ksh, posh, yash, zsh, bmake and make

artifactsdir=artifacts/artifacts-test
make_array=( bmake make )
shell_array=( ash bash dash ksh posh yash zsh )
for make_element in "${make_array[@]}"
do
	for shell_element in "${shell_array[@]}"
	do
		mkdir -p ${artifactsdir}/python/
		cp python/Makefile ${artifactsdir}/python/Makefile
		"$make_element" -C ${artifactsdir}/python/ SHELL="$shell_element" || rm -rf ${artifactsdir}
		"$make_element" -C ${artifactsdir}/python/ SHELL="$shell_element" clean || rm -rf ${artifactsdir}
		cp Makefile ${artifactsdir}/Makefile
		"$make_element" -C ${artifactsdir}/ SHELL="$shell_element" || rm -rf ${artifactsdir}
		"$make_element" -C ${artifactsdir}/ SHELL="$shell_element" artifacts/tex.tar || rm -rf ${artifactsdir}
		"$make_element" -C ${artifactsdir}/ SHELL="$shell_element" artifacts/tex-lint || rm -rf ${artifactsdir}
		"$make_element" -C ${artifactsdir}/ SHELL="$shell_element" artifacts/ms-server.pdf SERVER_URL=https://arxiv.org/e-print/2005.12660 || rm -rf ${artifactsdir}
		"$make_element" -C ${artifactsdir}/ SHELL="$shell_element" clean || rm -rf ${artifactsdir}
		rm -rf ${artifactsdir}/
	done
done
