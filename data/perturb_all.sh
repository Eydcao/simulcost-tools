for dir in */; do
  if [ -d "$dir" ]; then
    echo "Processing directory: $dir"
    for subdir in ${dir}*; do
        if [ -d "$subdir" ]; then
            echo "Processing subdirectory: ${subdir}"
            python perturb_inputs.py -i ${subdir}/input.cgyro -t ${subdir}/no_ion3_qn_input.cgyro -e ${subdir}/perturbation_error.npy -o ${subdir}/original_input.cgyro
        fi
    done
  fi
done
