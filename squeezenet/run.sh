for i in `seq 1 67`
do
  echo "File$i Inference Start";
  file_name=$(printf /home/hiroki11x/dl/models/img/dataset/File%d $i);\
  python sample.py $file_name
done
