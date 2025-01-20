<script lang="ts">
	import ThemeInputField from '../components/ThemeInputField.svelte';
	import ThemeButton from '../components/ThemeButton.svelte';
	import { activeComponent, data } from '../state.svelte';
	import { runModel, getDist } from '../routes/fetch.svelte';

	let { inpEventCb = null, value = $bindable() } = $props();

	let inpText: string = $state('');
	// let tokenProbMappings: { name: string, prob: number }[] = [];

	const randomInpText = () => {
		inpEventCb('alpha beta gamma delta eta zeta epsilon');
	};

	const runAndGetDist = async () => {
		await runModel(value);
		await getDist();
	}

	$effect(() => {
		$inspect(data.tokenProbMappings);
	})
</script>

<div>
	<ThemeInputField {inpEventCb} {value} maxlen={60} />
	<ThemeButton label="Generate" clickEventCb={runAndGetDist} />
	<ThemeButton label="Clear" clickEventCb={() => (value = '')} />
	<ThemeButton label="Load Example" clickEventCb={randomInpText} />
</div>
