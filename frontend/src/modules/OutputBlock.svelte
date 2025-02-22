<script lang="ts">
	import { QuestionCircleSolid } from 'flowbite-svelte-icons';

	import SideDrawer from '../components/SideDrawer.svelte';
	import ThemeInputSlider from '../components/ThemeInputSlider.svelte';
	import BarChart from '../dataviz/BarChart.svelte';
	import { activeComponent, data, global_state, input } from '../state.svelte';
	import ScatterChart from '../dataviz/ScatterChart.svelte';
	import HeatMap from '../dataviz/HeatMap.svelte';
	import MlpNeurons from '../dataviz/MLPNeurons.svelte';
	import DensityPlot from '../dataviz/DensityPlot.svelte';
	import {
		kSliderCallback,
		MLPPostCallback,
		MLPPreCallback,
		pSliderCallback,
		temperatureSliderCallback
	} from '../callbacks.svelte';
	import ThemeToggle from '../components/ThemeToggle.svelte';
	import { active } from 'd3';

	$effect(() => {
		$inspect(activeComponent);
		$inspect(data.tokenProbMappings);
	});

	// here true represent the top p and false mean k
	let topPorK = $state(false);
</script>

<SideDrawer bind:openState={global_state.ouputBlockState} width={'25vw'}>
	<div class="flex h-full w-full flex-col items-center justify-evenly pt-12 font-main-a">
		{#if activeComponent.name === 'MLP (in) Pre-activation' || activeComponent.name === 'GELU Activation'}
			<div class="flex flex-row items-center justify-evenly space-x-4">
				<label for="neuron">Neuron:</label>
				<input
					id="neuron"
					name="neuron"
					type="number"
					min="0"
					max="3072"
					bind:value={global_state.neuron}
					onchange={() => {
						if (activeComponent.name === "MLP (in) Pre-activation") {
							return MLPPreCallback();
						} else {
							return MLPPostCallback();
						}
					}}
					class="rounded-md border border-theme px-1 text-lg text-theme outline-none"
				/>
				<div class="flex flex-col">
					<span class="text-ti">3072</span>
					<span class="text-ti">0</span>
				</div>
			</div>
		{:else if activeComponent.name === 'Output Distribution'}
			<div class="relative w-full rounded-md bg-theme-g p-2 shadow-inner shadow-theme-g-alt">
				<a
					target="_blank"
					href="/read/control-parameter"
					title="Control Parameter"
					class="absolute end-1 top-1 text-theme"
				>
					<QuestionCircleSolid size={'sm'} />
				</a>
				<span
					class="mb-4 flex flex-row justify-around text-center text-sm font-extrabold uppercase text-theme underline"
				>
					Control Parameters
					<ThemeToggle
						bind:state={topPorK}
						style="z-50 text-ti-s"
						leftlabel="Top k"
						rightlabel="Top p"
					/>
				</span>
				<ThemeInputSlider
					label={'Temperature'}
					min={-2}
					max={2}
					step={0.1}
					changeEventCb={temperatureSliderCallback}
				/>
				<hr class="my-1 border border-theme-w" />
				{#if topPorK}
					<ThemeInputSlider
						label={'Top K'}
						min={1}
						max={10}
						step={1}
						changeEventCb={kSliderCallback}
					/>
				{:else}
					<ThemeInputSlider
						label={'Top P'}
						min={0}
						max={1}
						step={0.05}
						changeEventCb={pSliderCallback}
					/>
				{/if}
			</div>
		{/if}
		<hr class="w-full border border-theme" />
		<h1 class="text-md my-2 text-center font-extrabold uppercase text-theme">
			{activeComponent.name}
		</h1>
		<div
			class="flex min-h-[22rem] w-full flex-col items-center justify-evenly rounded-md border bg-theme-g p-3 shadow-inner shadow-gray-400"
		>
			<div class="chart-container w-full text-right text-ti font-light">
				{#if input.text === ''}
					Enter Something.
				{:else if activeComponent.name === 'Generate' || activeComponent.name === 'Output Distribution'}
					<BarChart tokens={data.tokenProbMappings} />
				{:else if activeComponent.name === 'Token Embedding' || activeComponent.name === 'Positional Embedding'}
					<ScatterChart data={global_state.embed_output} />
				{:else if activeComponent.name === 'Attention Pattern'}
					<HeatMap data={global_state.attn_patterns[global_state.active_head]} vmax="#03045E" />
				{:else if activeComponent.name === 'MLP (in) Pre-activation' || activeComponent.name === 'GELU Activation'}
					<MlpNeurons data={global_state.data} />
				{:else if activeComponent.name === 'LN1' || activeComponent.name === 'LN2' || activeComponent.name === 'LN Final'}
					<DensityPlot pre={global_state.ln_pre} post={global_state.ln_post} />
				{/if}
			</div>
		</div>
		<span class="my-2 font-bold text-theme"
			>Next Token : <span class="rounded-md bg-theme p-1 px-2 font-light text-theme-w"
				>{global_state.next_token}</span
			></span
		>
	</div>
</SideDrawer>

<style lang="css">
	.chart-container {
		height: 100%;
	}
</style>
