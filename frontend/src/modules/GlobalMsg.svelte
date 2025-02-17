<script lang="ts">
	import Message from '../components/Message.svelte';
	import { global_state } from '../state.svelte';

	let infoInt: any = null;

	$effect(() => {
		if (global_state.info && !infoInt) {
			infoInt = setTimeout(() => {
				global_state.info = null;
				infoInt = null;
				// console.log("Timeout for message")
			}, 4000);
		}
	});

	const closeMsg = () => {
		if (infoInt) {
			clearInterval(infoInt);
			global_state.info = null;
			infoInt = null;
		}
	};
</script>

{#if global_state.info}
	<Message message={global_state.info?.message} type={global_state.info?.code}>
		<button onclick={() => closeMsg()} class="ml-5 text-sm hover:scale-125">X</button>
	</Message>
{/if}
