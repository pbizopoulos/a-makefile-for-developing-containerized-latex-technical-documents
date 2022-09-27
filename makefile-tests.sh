#!/bin/bash
set -o errexit

# Requirements: ash, bash, dash, ksh, posh, yash, zsh, bmake and make

artifacts_dir=artifacts/artifacts-test
make_array=( bmake make )
shell_array=( ash bash dash ksh posh yash zsh )
for make_element in "${make_array[@]}"
do
	for shell_element in "${shell_array[@]}"
	do
		mkdir -p ${artifacts_dir}/python/
		cp python/Makefile ${artifacts_dir}/python/Makefile
		"$make_element" -C ${artifacts_dir}/python/ SHELL="$shell_element" || rm -rf ${artifacts_dir}
		"$make_element" -C ${artifacts_dir}/python/ SHELL="$shell_element" clean || rm -rf ${artifacts_dir}
		cp Makefile ${artifacts_dir}/Makefile
		"$make_element" -C ${artifacts_dir}/ SHELL="$shell_element" || rm -rf ${artifacts_dir}
		"$make_element" -C ${artifacts_dir}/ SHELL="$shell_element" artifacts/tex.tar || rm -rf ${artifacts_dir}
		"$make_element" -C ${artifacts_dir}/ SHELL="$shell_element" artifacts/tex-lint || rm -rf ${artifacts_dir}
		"$make_element" -C ${artifacts_dir}/ SHELL="$shell_element" artifacts/ms-server.pdf SERVER_URL=https://arxiv.org/e-print/2005.12660 || rm -rf ${artifacts_dir}
		"$make_element" -C ${artifacts_dir}/ SHELL="$shell_element" clean || rm -rf ${artifacts_dir}
		rm -rf ${artifacts_dir}/
	done
done
