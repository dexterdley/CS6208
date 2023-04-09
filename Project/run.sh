#python CS6208_project.py --epochs=100 --dataset="x5_x5" --expt_name="x5_x5_focal" --gamma=2
python CS6208_project.py --epochs=100 --dataset="x68_x68" --expt_name="x68_x68_focal" --gamma=2
python CS6208_project.py --epochs=100 --dataset="x5_x68" --expt_name="x5_x68_focal" --gamma=2
python CS6208_project.py --epochs=100 --dataset="x68_x5" --expt_name="x68_x5_focal" --gamma=2

#python CS6208_project.py --epochs=100 --dataset="x5_x5" --expt_name="x5_x5_maxent_mu" --gamma=2 --constraints=1
python CS6208_project.py --epochs=100 --dataset="x68_x68" --expt_name="x68_x68_maxent_mu" --gamma=2 --constraints=1
python CS6208_project.py --epochs=100 --dataset="x5_x68" --expt_name="x5_x68_maxent_mu" --gamma=2 --constraints=1
python CS6208_project.py --epochs=100 --dataset="x68_x5" --expt_name="x68_x5_maxent_mu" --gamma=2 --constraints=1