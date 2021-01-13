import os
import pandas as pd

class CSVLogger():
    def __init__(self, filename, metrics, log_interval):
        self.filename = filename
        self.log_interval = log_interval
        self.metrics = metrics
        self.current_step = 0

        # Get CSV file header, if necessary.
        header = self._get_header() if not os.path.exists(filename) else None

        # Write header to the output file, if necessary.
        if header:
            self.out_file = open(self.filename, 'a')
            self.out_file.write(header)
            self.out_file.close()

    def set_step(self, step):
        self.current_step = step

    def append_metrics(self, metric_value_dict):
        metrics_dict = {m: value for m, value in metric_value_dict.items() if m in self.metrics}
        self.current_step += self.log_interval
        self._save_metrics(metrics_dict)

    def _save_metrics(self, metrics_dict):
        # Open file in append mode.
        self.out_file = open(self.filename, 'a')

        # Get CSV metrics line.
        out = '\n' + str(self.current_step)
        for m in self.metrics: out += ',' + str(metrics_dict[m])

        # Write line and close file.
        self.out_file.write(out)
        self.out_file.close()
        
    def _get_header(self):
        out = 'step'
        for m in self.metrics: out += ',' + m 
        
        return out

def csv_logger(metrics, log_interval, exp_dir):
    train_metrics = ['train_loss'] + [x+'_loss' for x in metrics]
    logger = CSVLogger('{:s}/{:s}'.format(exp_dir, 'training.log'), train_metrics, log_interval)
    return logger

def save_parms(output_directory, args, loss_weights=None):
    """ Save argument parameters to log file.

        Keyword arguments:
        --------------------------------------------------
        output_directory(string) -- Output directory
        args(Namespace) -- Argparse arguments
        loss_weights (dictionary) -- Weights for loss function.
    """
    
    with open(os.path.join(output_directory, 'log.txt'), 'w+') as f:
        # Save parameters.
        f.write('Experiment Parameters\n')
        f.write('=========================')
        f.write('\n')
        for arg in vars(args):
            f.write("{}: {}\n".format(arg, getattr(args, arg)))

        f.write('\n\nExperiment Loss Weights \n')
        f.write('=========================')
        f.write('\n')
        for key, coef in loss_weights.items():
            f.write("{}: {}\n".format(key, coef))
        f.write('------------------------- \n\n')

def save_metrics(output_directory, metrics):
    """ Save metrics to output file.

        Keyword arguments:
        --------------------------------------------------
        output_directory(string) -- Output directory
        metrics (dictionary) -- Weights for loss function.
    """

    with open(os.path.join(output_directory, 'metrics.txt'), 'w+') as f:
        f.write('Metrics results\n')
        f.write('=========================\n\n')

        f.write("Number of images evaluated: %d\n" % len(metrics['files']))
        f.write("Median MSE: %0.5f\n" % np.median(metrics['MSE']))
        f.write("Mean MSE: %0.5f\n" % np.mean(metrics['MSE']))
        f.write("Std MSE: %0.5f\n" % np.std(metrics['MSE']))

        f.write("Median MAE: %0.5f\n" % np.median(metrics['MAE']))
        f.write("Mean MAE: %0.5f\n" % np.mean(metrics['MAE']))
        f.write("Std MAE: %0.5f\n" % np.std(metrics['MAE']))
        f.write('------------------------- \n\n')

    pd.DataFrame.from_dict(metrics).to_csv(os.path.join(output_directory, 'out.csv'))