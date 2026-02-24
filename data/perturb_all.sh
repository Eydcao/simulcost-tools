for dir in */; do
  if [ -d "$dir" ]; then
    echo "Processing directory: $dir"
    for subdir in ${dir}*; do
        if [ -d "$subdir" ]; then
            echo "Processing subdirectory: ${subdir}"
            python perturb_inputs.py -i ${subdir}/input.cgyro -t ${subdir}/original_input.cgyro -e ${subdir}/perturbation_error.npy
        fi
    done
  fi
done
