import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split 

class LinReg:
    
    def true_relationship(self):
        """
        Return string with the true relationship between target and features
        """
        
        return f'y = {self.x_0_coef} + {self.x_1_coef}x + {self.x_1_sq_coef}$x^2$ + {self.x_2_coef}$x_2$ + noise'

    def corr_var(self, x, rho, sigma):
        """
        Create a variable correlated with variable x
        """
        sigma_x = np.std(x)

        samples = x.shape[-1]
        x_2 = rho * sigma/sigma_x * x + (1 - rho**2)**0.5 * sigma * np.random.normal(0, 1, samples)

        return x_2

    def gen_data(self, samples=200, min_x=0, max_x=10, x_0_coef=100, x_1_coef=2, x_1_sq_coef=3, x_2_coef=0.25, rho=0.5, sigma_mult=0.5, seed=42):
        """
        Generate a dataset
        """
        # Save the configuration
        self.samples = samples
        self.max_x, self.min_x = max_x, min_x
        self.x_0_coef, self.x_1_coef, self.x_1_sq_coef, self.x_2_coef = x_0_coef, x_1_coef, x_1_sq_coef, x_2_coef
        self.rho, self.sigma_mult = rho, sigma_mult
        self.seed = seed

        # Set the seed
        np.random.seed(seed)
        
        # Create x_1
        x_1 = np.linspace(min_x, max_x, samples)
        sigma_1 = np.std(x_1)

        # Create correlated  (with correlation rho) x_2
        sigma_2 = sigma_mult * sigma_1

        x_2 = self.corr_var(x_1, rho, sigma_2)
        
        # Create y with x_1 terms
        y = x_0_coef + x_1_coef * x_1 + x_1_sq_coef * x_1**2 

        # Augment y with x_2 term
        y = y + x_2_coef * x_2

        # Add noise to y
        y_noise = np.random.normal(0, (y.max()-y.min()) * .01, samples)
        y = y + y_noise

        vars = { "x_1": x_1,
                 "x_2": x_2,
                 "y": y
               }
        df = pd.DataFrame( vars )
        
        return df
    
    def plot(self, model, y, X, feature_names=[], showEquation=True, showTrue=True, title=None, visible=True):
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate R-squared
        r_squared = r2_score(y, y_pred)

        # Horizontal axis is first feature (ignoring other features, just in the plot)
        x_1 = X[:,0]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot of actual training data: x_1, y
        _= ax.scatter(x_1, y, color='blue', alpha=0.5, label='Data points')
        
        # Line plot fitted values (prediction)
        # Re-order the data in increasing value of x_1
        # - this is only so that the fitted values appear as a line
        sort_idx = x_1.argsort()
        x_1_sort, y_pred_sort = x_1[sort_idx], y_pred[sort_idx]
        _= ax.plot(x_1_sort, y_pred_sort, color='red', label='Regression')

        # Enumerate features
        features_text = ", ".join(feature_names)
        
        # Set labels and title
        _=  ax.set_xlabel('$x_1$')
        _= ax.set_ylabel('y')
        
        title_string = title if title is not None else ""
        title_string += f'Features: {features_text}\nR-squared: {r_squared:.4f}'
        
        _= ax.set_title(title_string)
        _= ax.legend()

        # Add text describing the true relationship
        if showTrue:
            _= ax.text(0.05, 0.75, f'True relationship: {self.true_relationship()}', 
                       transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

        # Add text describing the model
        if showEquation:
            equation_terms =  [ f"{model.intercept_:.2f}" ]
            for i in range(X.shape[-1]):
                equation_terms.append(f"{model.coef_[i]:.2f}{feature_names[i]}")

            equation = f'Prediction: y = {" + ".join(equation_terms)} '
            _= ax.text(0.05, 0.85, equation, transform=plt.gca().transAxes, 
                       verticalalignment='top', fontsize=10)

        _= plt.tight_layout()

        if not visible:
            _= plt.close(fig)
        else:
            _= plt.show()

        return fig, ax

    def plot_pred_vs_true(self, model, y, X,  feature_names=[], showEquation=True,  showTrue=False, title=None, visible=True):
        # Make predictions
        y_pred = model.predict(X)

        # Calculate R-squared
        r_squared = r2_score(y, y_pred)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Axes: True y, Predicted y
        _= ax.set_xlabel('True Values')
        _= ax.set_ylabel('Predictions')
        _= ax.scatter(y, y_pred, c='g', alpha=0.5)
        _= ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='True=Predicted')

        # Add text describing the true relationship
        if showTrue:
            _= ax.text(0.05, 0.75, f'True relationship: {self.true_relationship()}', 
                       transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

        # Add text describing the model
        if showEquation:
            equation_terms =  [ f"{model.intercept_:.2f}" ]
            for i in range(X.shape[-1]):
                equation_terms.append(f"{model.coef_[i]:.2f}{feature_names[i]}")

            equation = f'Prediction: y = {" + ".join(equation_terms)} '
            _= ax.text(0.05, 0.85, equation, transform=plt.gca().transAxes, 
                       verticalalignment='top', fontsize=10)
            
        # Create the title
        title_string = title if title is not None else ""
        title_string += f'R-squared: {r_squared:.4f}'
        
        _= ax.set_title(title_string)
        _= ax.legend()

        _= plt.tight_layout()

        if not visible:
            _= plt.close(fig)
        else:
            _= plt.show()

        return fig, ax

    # Create figures for the lecture
    def createFigs(self, visible=False):
        # Some options to configure the data
        rho, sigma_mult = 0, 20

        # Generate the data
        df = self.gen_data(samples=300, rho=rho, sigma_mult=sigma_mult)

        # Separate the features from the target
        X = df[ ["x_1", "x_2"]].to_numpy()
        y = df[ "y"].to_numpy()

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                           random_state=self.seed,  
                                           test_size=0.2,  
                                           shuffle=True) 

        # Create the scatter plot of the examples
        fig_data, ax_data = plt.subplots(figsize=(10, 6))

        _= ax_data.scatter(X_train[:,0], y_train, color='blue', alpha=0.5, label='Data points')


        # Set labels and title
        _= ax_data.set_xlabel('$x_1$')
        _= ax_data.set_ylabel('$y$')
        _= ax_data.legend()

        _= plt.tight_layout()

        if not visible:
            _= plt.close(fig_data)

    
        # Create plot of fitting to one variable
        from sklearn.linear_model import LinearRegression

        # Create and fit the linear regression model
        model = LinearRegression()

        # Note: we access the single feature as X_train[:, [9]] rather than X_train[:, 0] to ensure the result is 2D
        # - with a singlteon final dimension, as required by sklearn
        _= model.fit(X_train[:,[0] ], y_train)
    
        fig_x1, ax_x1 = self.plot(model, y_test, X_test[:,[0] ], feature_names=["$x_1$"],
                                  title="Linear Regression\nOut of sample (test data)\n",
                                  showTrue=False,
                                  visible=visible
                                  )

    
        if not visible:
            _= plt.close(fig_x1)

    
        # Create plot of fit vs x_1, x_1^2
        # Add an x_1^2 column to X_train AND X_test
        X_train_with_sq = np.column_stack( (X_train[:, 0], X_train[:, 0]**2) )
        X_test_with_sq = np.column_stack( (X_test[:, 0], X_test[:, 0]**2) )

        # Fit the model
        model = LinearRegression()
        _= model.fit(X_train_with_sq, y_train)

        fig_x1_x1sq, ax_x1_x1sq = self.plot(model, y_test, X_test_with_sq, feature_names=["$x_1$", "$x_1^2$"],
                                            title="Linear Regression\nOut of sample (test data)\n",
                                            showTrue=False,
                                            visible=visible
                                            )


        if not visible:
            _= plt.close(fig_x1_x1sq)
        
        return (fig_data, fig_x1, fig_x1_x1sq)
