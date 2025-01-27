<script lang="ts">
	import ThemeInputField from '../components/ThemeInputField.svelte';
	import ThemeButton from '../components/ThemeButton.svelte';
	import { activeComponent, data, global_state, input } from '../state.svelte';
	import { loadModel, runModel, getDist } from '../routes/fetch.svelte';
	import { active } from 'd3';

	let { inpEventCb = null, value = $bindable() } = $props();

	const randomInpText = () => {
		inpEventCb('alpha beta gamma delta eta zeta epsilon');
	};

	const runAndGetDist = async () => {
		if (global_state.isModelLoaded === false) {
			await loadModel();
		}
		console.log("is model loaded? " + global_state.isModelLoaded);

		// will have to test this more rigorously.
		try {
			await runModel(value);
		} catch {
			await loadModel();
			await runModel(value);
		}
		await getDist();

		activeComponent.name = "Output Distribution";
		input.isChanged = false;
	};

	$effect(() => {
		$inspect(data.tokenProbMappings);
	});
</script>

<div>
	<ThemeInputField {inpEventCb} {value} maxlen={60} />
	<ThemeButton label="Generate" clickEventCb={runAndGetDist} />
	<ThemeButton label="Clear" clickEventCb={() => (value = '')} />
	<ThemeButton label="Load Example" clickEventCb={randomInpText} />
</div>
