echo "Enter one of the following numbers:"
echo ""
echo " 1) setup virtual environment mapinwild"
echo " 2) activate virtual environment mapinwild"
echo ""
read OPTION
echo ""


if [[ OPTION -eq 1 ]]
then    
    conda env create -f environment.yml    
    conda activate mapinwild
    earthengine authenticate
fi

if [[ OPTION -eq 2 ]]
then
    conda activate mapinwild
fi

CWD=$(pwd)
export PYTHONPATH=$CWD:$PYTHONPATH
export JUPYTER_PATH=$CWD:$JUPYTER_PATH

