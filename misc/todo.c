#include <math.h>
/* model_params array indeces:
	'c_ee',		0
    'c_ei', 	1
    'c_ie', 	2
    'c_ii', 	3
    'tau_e', 	4
    'tau_i', 	5
    'a_e', 		6
    'b_e', 		7
    'c_e', 		8
    'a_i', 		9
    'b_i', 		10
    'c_i', 		11
    'r_e', 		12
    'r_i', 		13
    'k_e', 		14
    'k_i', 		15
    'P', 		16
    'Q', 		17
    'theta_e', 	18
    'theta_i', 	19
    'alpha_e', 	20
    'alpha_i'	21
*/

void dfun(	double* state_variables, 
			double* coupling, 
			double* local_coupling, 
			int n_nodes,
			int n_modes,
			double* model_params, // order to be derived from the ui_configurable_parameters
			double* derivative
			){
        
	for(int node_it = 0; node_it < n_nodes; node_it ++){
			double E, I, c_0, lc_0, lc_1, x_e, x_i, s_e, s_i, dE, dI;

			E = state_variables[0][node_it];
    	    I = state_variables[1][node_it];

    	    c_0 = coupling[0][node_it][mode_it];

    	    lc_0 = local_coupling[node_it] * E;
    	    lc_1 = local_coupling[node_it] * I;

    	    x_e = model_params[20] * (model_params[0] * E - model_params[1] * I + model_params[16]  - model_params[18] +  c_0 + lc_0 + lc_1);
    	    x_i = model_params[21] * (model_params[2] * E - model_params[3] * I + model_params[17]  - model_params[19] + lc_0 + lc_1);

    	    s_e = model_params[8] / (1.0 + numpy.exp(-model_params[6] * (x_e - model_params[7])));
    	    s_i = model_params[11] / (1.0 + numpy.exp(-model_params[9] * (x_i - model_params[10])));

    	    dE = (-E + (model_params[14] - model_params[12] * E) * s_e) / model_params[4];
    	    dI = (-I + (model_params[15] - model_params[13] * I) * s_i) / model_params[5];

    	    derivative[0][node_it] = dE;
    	    derivative[1][node_it] = dI;
	}
}
