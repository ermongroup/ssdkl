for dataset in skillcraft parkinsons elevators protein blog ctslice buzz electric; do
    echo $dataset
    batch_size=64
    if [ "$dataset" = buzz ] ; then
        epochs=10
    elif [ "$dataset" = electric ] ; then
        epochs=10
    else
        epochs=100
    fi

    echo batch_size $batch_size

    echo epochs $epochs

    export CUDA_VISIBLE_DEVICES="1"

    python train_vae.py --gpu 1 --batch_size $batch_size --dataset $dataset --z_dim 2 --epochs $epochs &
done
