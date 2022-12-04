def add_weight_decay(grads_and_vars, weight_decay=0.0):
    if weight_decay != 0.0:
        for (gradient, variable) in grads_and_vars:
            gradient.assign_add(weight_decay * variable)