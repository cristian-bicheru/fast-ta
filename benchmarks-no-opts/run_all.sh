for f in *.py
do
	echo "Benchmarking ${f:0:(-3)}"
	python3.7 $f
done
echo "done"
