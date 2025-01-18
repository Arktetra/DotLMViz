<script lang="ts">
	import ThemeInputField from '../components/ThemeInputField.svelte';
	import ThemeButton from '../components/ThemeButton.svelte';

	let { inpEventCb = null, btnEventCb, value = $bindable() } = $props();

	const randomInpText = () => {
		inpEventCb('alpha beta gamma delta eta zeta epsilon');
	};
</script>

<div>
	<ThemeInputField {inpEventCb} {value} maxlen={60} />
	<ThemeButton label="Generate" clickEventCb={btnEventCb} />
	<ThemeButton label="Clear" clickEventCb={() => (value = '')} />
	<ThemeButton label="Load Example" clickEventCb={randomInpText} />
</div>
