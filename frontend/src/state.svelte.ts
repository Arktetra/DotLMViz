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

export const input = $state({
	isChanged: false,
	text: ""
});

// All the global state here
// #note: currently this is not to be used, left to configure
export const global_state = $state<{
	isModelLoaded: boolean,
	data: any[],
	tokens: any[],
	embed_output: ScatterPlotData,
	ln_pre: [],
	ln_post: [],
	attn_patterns: HeatMapData[],	// contains attn pattern for all heads
	active_head: number,
	active_block: number,
	neuron: number,
	viewMode: boolean
}>({
	isModelLoaded: false,
	data: [],
	tokens: [],
	embed_output: [],
	ln_pre: [],
	ln_post: [],
	attn_patterns: [],
	active_head: 0,
	active_block: 0,
	neuron: 0,
	viewMode: false
});
