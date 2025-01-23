export const activeComponent = $state({
	name: 'Output Distribution'
});

export const data = $state({
	tokenProbMappings: []
});

// All the model information name, embed_size, ...
export const active_model = $state({
	model_name: 'gpt2-small'
});

// All the global state here
// #note: currently this is not to be used, left to configure
export const global_state = $state({
	isModelLoaded: false,
	data: []
});
