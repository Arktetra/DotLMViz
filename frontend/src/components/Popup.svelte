<script lang="ts">
	import { CloseOutline } from 'flowbite-svelte-icons';
	import FullScreenOverlay from './FullScreenOverlay.svelte';

	const { onCloseCb, style = '', children = null } = $props();

	const onClose = () => {
		if (onCloseCb) onCloseCb();
	};
</script>

<FullScreenOverlay clickEvent={onCloseCb}>
	<div
		on:click|stopPropagation
		class="relative z-20 flex min-h-[15rem] min-w-[15rem] flex-col items-center justify-center rounded-md bg-theme p-2 text-theme-w {style}"
	>
		<button
			on:click={() => onClose()}
			class=" absolute -right-2 -top-2 cursor-pointer transition-transform duration-300 hover:scale-105"
		>
			<CloseOutline class="h-5 w-5 rounded-xl bg-theme-w text-theme-r" />
		</button>

		{#if children}
			{@render children()}
		{/if}
	</div>
</FullScreenOverlay>
