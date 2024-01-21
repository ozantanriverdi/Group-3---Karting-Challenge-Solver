import importlib
import json
import os
from plotter.interactive_training_plotter import InteractiveTrainingPlotter
from utils.enums import PlottingSettingEnum
class HTMLPlotter: 
    def __init__(self):
        self.plotter = InteractiveTrainingPlotter()
        pass

    def update_html_file(self):
        log_folder = "logger/logs"    
        result_file = "plotter/training_results.html"

        standard_hyperparameters = {}
        spec = importlib.util.spec_from_file_location("hyperparameters", "configs/hyperparameters.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        standard_hyperparameters = module.hyperparameters

        standard_run_configs = {}
        spec = importlib.util.spec_from_file_location("run_config", "configs/run_config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        standard_run_configs = module.config
        print(standard_run_configs)

        html_content = ""
        html_content += "<html>"
        html_content += "<head>"
        html_content += "<style>"
        html_content += ".plot { display: block; }"  # Hide the plots by default
        html_content += ".shared-plt { display: none; }"  # Hide the plots by default
        html_content += "body { font-family: Arial, sans-serif; }"
        html_content += "span.diff-param { color: green; }"
        html_content += "</style>"
        html_content += "</head>"
        html_content += "<body>"
        html_content += "<div style='position: fixed; '><button onclick='togglePlots()'>Toggle Plots</button></div>"  # Button to toggle plot visibility
        html_content += "<script>"
        html_content += "function togglePlots() {"
        html_content += "  var plots = document.getElementsByClassName('plot');"
        html_content += "  var shared_plots = document.getElementsByClassName('shared-plt');"
        html_content += "  for (var i = 0; i < plots.length; i++) {"
        html_content += "    if (plots[i].style.display === 'none') {"
        html_content += "      plots[i].style.display = 'block';"  # Show the plot
        html_content += "    } else {"
        html_content += "      plots[i].style.display = 'none';"  # Hide the plot
        html_content += "    }"
        html_content += "  }"
        html_content += "  for (var i = 0; i < shared_plots.length; i++) {"
        html_content += "    if (shared_plots[i].style.display === 'none') {"
        html_content += "      shared_plots[i].style.display = 'block';"  # Show the plot
        html_content += "    } else {"
        html_content += "      shared_plots[i].style.display = 'none';"  # Hide the plot
        html_content += "    }"
        html_content += "  }"
        html_content += "}"
        html_content += "</script>"

        grouped_logs = {}

        for filename in os.listdir(log_folder):
            if filename.endswith(".json"):
                training_name = os.path.splitext(filename)[0]
                plot_html = self.plotter.plot_previous_training(training_name, as_html_plot=True)
                
                # Load hyperparameters and run_configs from JSON
                with open(os.path.join(log_folder, filename), "r") as json_file:
                    data = json.load(json_file)
                    hyperparameters = data.get("hyperparameters", {})
                    run_configs = data.get("run_configs", {})
                  
                    seed = None
                    if 'seed' in run_configs:
                        seed = run_configs['seed']
                        del run_configs['seed']

                    # Create a key tuple from hyperparameters and run_configs
                    key = (tuple(hyperparameters.items()), tuple(run_configs.items()))
 
                    # Add the log to the corresponding key in the grouped_logs dictionary
                    if key in grouped_logs:
                        grouped_logs[key].append((training_name, seed))
                    else:
                        grouped_logs[key] = [(training_name, seed)]


        for key, trainings in grouped_logs.items():
            # Extract hyperparameters and run_configs from the key
            hyperparameters = dict(key[0])
            run_configs = dict(key[1])
        
            # Start a new row for each grouped_logs
            html_content += "<div style='display: flex;'>"

            # Create the column for hyperparameters and run_configs
            html_content += "<div style='width: 30%; padding-left: 20px; padding-right: 20px;'>"
            html_content += "<h2>Hyperparameters:</h2>"
            for param, value in hyperparameters.items():
                # Highlight the parameters that differ from the standard hyperparameters
                if param in standard_hyperparameters and value != standard_hyperparameters[param]:
                    html_content += "<p><span style='color: green;'>{}</span>: {} (standard: {})</p>".format(param, value, standard_hyperparameters[param])
                else:
                    html_content += "<p>{}: {}</p>".format(param, value)
            html_content += "<h2>Run Configs:</h2>"
            for param, value in run_configs.items():
                # Highlight the parameters that differ from the standard run_configs
                if param in standard_run_configs and value != standard_run_configs[param]:
                    html_content += "<p><span style='color: green;'>{}</span>: {} (standard: {})</p>".format(param, value, standard_run_configs[param])
                else:
                    html_content += "<p>{}: {}</p>".format(param, value)
            html_content += "</div>"
            html_content += "<hr style='margin-right: 20px'>"
            
            html_content += "<div class='plot' style='width: 70%; display: block'>"
            for count, (training_name, training_seed) in enumerate(trainings):
                training_names = [training[0] for training in trainings]
                plot_html = self.plotter.plot_previous_training(training_name, as_html_plot=True)
                html_content += "<h2>Plot for log: {}</h2>".format(training_name)
                html_content += "<p>Seed: {}</p>".format(training_seed if training_seed else 'None')
                html_content += plot_html
                if count < len(trainings) - 1:
                    html_content += "<hr>"
            html_content += "</div>"
            html_content += "<div class='shared-plt' style='width: 70%; display: none'>"
            html_content += "<h2>Plot for logs</h2>"
            html_content += "<p>{}</p>".format(training_names)

            print(training_names)
            shared_plot = self.plotter.plot_multiple_trainings_with_same_config([training_names], PlottingSettingEnum.REWARDS, as_html_plot=True)
            html_content += shared_plot
            html_content += "</div>"

            # Close the row
            html_content += "</div>"
            html_content += "<hr>"

        # End the HTML content
        html_content += "</body>"
        html_content += "</html>"

        with open(result_file, "w") as html_file:
            html_file.write(html_content)