<script lang="ts">
	import ThemeInputField from '../components/ThemeInputField.svelte';
	import ThemeButton from '../components/ThemeButton.svelte';
	import { data } from '../state.svelte';
	import { outputCallback } from '../callbacks.svelte';

	let { inpEventCb = null, value = $bindable() } = $props();

	const randomInpText = () => {
		inpEventCb('alpha beta gamma delta eta zeta epsilon');
	};

	$effect(() => {
		$inspect(data.tokenProbMappings);
	});
</script>

<div class="flex flex-row justify-between items-center space-x-5">
	<ThemeInputField {inpEventCb} {value} maxlen={60} />
	<ThemeButton label="Generate" clickEventCb={outputCallback} />
	<ThemeButton label="Clear" clickEventCb={() => inpEventCb('')} />
	<ThemeButton label="Example" clickEventCb={randomInpText} />
</div>
